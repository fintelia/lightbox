#![feature(underscore_imports)]

extern crate failure;
extern crate gfx_core;
extern crate image;
extern crate obj;

#[cfg(feature = "headless")]
extern crate gfx_device_gl;
#[cfg(feature = "headless")]
extern crate glutin;

#[macro_use]
extern crate gfx;
#[macro_use]
extern crate gfx_shader_watch;

use failure::Error;
use gfx::memory::{Bind, Usage};
use gfx::state::Rasterizer;
use gfx::texture::{AaMode, Kind};
use gfx::traits::FactoryExt;
use gfx::{CommandBuffer, Device, Encoder, Factory, IndexBuffer, Primitive, Resources, Slice};
use gfx_core::format::{ChannelType, DepthStencil, Formatted, Srgba8};
use gfx_core::handle::{Buffer, DepthStencilView, RenderTargetView};
use gfx_core::memory::Typed;
use gfx_shader_watch::PsoCell;
use image::RgbaImage;
use obj::{Mtl, Obj, SimplePolygon};
use std::collections::HashMap;
use std::sync::Arc;

gfx_defines!{
    vertex Vertex {
        position: [f32; 3] = "vPosition",
        texcoord: [f32; 3] = "vTexcoord",
        normal: [f32; 3] = "vNormal",
    }
}
gfx_pipeline!( pipe {
    vertices: gfx::VertexBuffer<Vertex> = (),
    texture: gfx::TextureSampler<[f32; 4]> = "uTexture",

    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",

    color: gfx::RenderTarget<Srgba8> = "OutColor",
    depth: gfx::DepthTarget<DepthStencil> = gfx::preset::depth::PASS_WRITE,
});

pub struct Model<R: Resources> {
    slice: Slice<R>,
    data: pipe::Data<R>,
    min: [f32; 3],
    max: [f32; 3],
}
impl<R: Resources> Model<R> {
    pub fn from_obj<'a, F: Factory<R>>(
        lightbox: &mut Lightbox<R, F>,
        mut model: Obj<'a, SimplePolygon>,
        mtl: Option<Mtl>,
        textures: HashMap<String, Arc<RgbaImage>>,
    ) -> Result<Self, Error> {
        if model.normal.is_empty() {
            model.normal.push([0.0, 0.0, 0.0]);
        }
        if model.texture.is_empty() {
            model.texture.push([0.0, 0.0]);
        }

        let textures: Vec<(String, _)> = textures.into_iter().collect();
        let texture_indices: HashMap<String, usize> = textures
            .iter()
            .enumerate()
            .map(|(i, (ref name, _))| (name.clone(), i))
            .collect();

        let width = textures.iter().map(|m| m.1.width()).max().unwrap_or(1) as usize;
        let height = textures.iter().map(|m| m.1.height()).max().unwrap_or(1) as usize;

        let mut texture_data = vec![255u8; 4 * width * height * textures.len().max(1)];
        for (i, (_, texture)) in textures.iter().enumerate() {
            let w = texture.width() as usize;
            let h = texture.height() as usize;
            if w > 0 && h > 0 {
                for y in 0..h {
                    let slice: &[u8] = &*texture;
                    assert_eq!(slice.len(), w * h * 4);
                    texture_data[4 * (y * width + i * width * height)..][..(w * 4)]
                        .copy_from_slice(&slice[4 * (y * w)..][..(w * 4)]);
                }
            }
        }
        let texture_data_slices: Vec<_> = (0..textures.len().max(1))
            .map(|i| &texture_data[i * width * height * 4..][..width * height * 4])
            .collect();;

        let texture = lightbox
            .factory
            .create_texture_immutable_u8::<(gfx_core::format::R8_G8_B8_A8, gfx_core::format::Srgb)>(
                gfx::texture::Kind::D2Array(
                    width as u16,
                    height as u16,
                    textures.len().max(1) as u16,
                    gfx::texture::AaMode::Single,
                ),
                gfx::texture::Mipmap::Provided,
                &texture_data_slices[..],
            )?.1;

        let sampler = lightbox
            .factory
            .create_sampler(gfx::texture::SamplerInfo::new(
                gfx::texture::FilterMethod::Bilinear,
                gfx::texture::WrapMode::Tile,
            ));

        let material_indices: HashMap<String, usize> = match mtl {
            Some(ref mtl) => mtl
                .materials
                .iter()
                .enumerate()
                .map(|(i, m)| (m.name.clone(), i))
                .collect(),
            None => HashMap::new(),
        };

        let mut vertices = Vec::new();
        for object in &model.objects {
            for group in &object.groups {
                let (wscale, hscale, layer) = match group
                    .material
                    .as_ref()
                    .and_then(|ref m| material_indices.get(&m.name))
                    .map(|&i| &mtl.as_ref().unwrap().materials[i])
                    .and_then(|ref m| texture_indices.get(&m.name))
                    .and_then(|&i| Some((i, textures[i].clone())))
                {
                    Some((i, m)) => (
                        m.1.width() as f32 / width as f32,
                        m.1.height() as f32 / height as f32,
                        i as f32,
                    ),
                    None => (0.0, 0.0, 0.0),
                };

                for poly in &group.polys {
                    for i in 0..poly.len().checked_sub(2).unwrap() {
                        for &j in [0, i + 1, i + 2].iter() {
                            let tc = model.texture[poly[j].1.unwrap_or(0)];
                            vertices.push(Vertex {
                                position: model.position[poly[j].0],
                                texcoord: [tc[0] * wscale, tc[1] * hscale, layer],
                                normal: model.normal[poly[j].2.unwrap_or(0)],
                            });
                        }
                    }
                }
            }
        }

        let mut min = [std::f32::INFINITY; 3];
        let mut max = [std::f32::NEG_INFINITY; 3];
        for v in &vertices {
            for i in 0..3 {
                if min[i] > v.position[i] {
                    min[i] = v.position[i];
                }
                if max[i] < v.position[i] {
                    max[i] = v.position[i];
                }
            }
        }

        let (vertices, slice) = lightbox
            .factory
            .create_vertex_buffer_with_slice(&vertices, IndexBuffer::Auto);

        Ok(Self {
            slice,
            data: pipe::Data {
                color: lightbox.color_target.clone(),
                depth: lightbox.depth_target.clone(),
                model_view_projection: [[-12.0; 4]; 4],
                vertices,
                texture: (texture, sampler),
            },
            min,
            max,
        })
    }
}

pub struct Lightbox<R, F>
where
    R: Resources,
    F: Factory<R>,
{
    width: u16,
    height: u16,

    factory: F,

    download_buffer: Buffer<R, u8>,

    pso_cell: Box<PsoCell<R, F, pipe::Init<'static>>>,
    color_target: RenderTargetView<R, Srgba8>,
    depth_target: DepthStencilView<R, DepthStencil>,
}

#[cfg(feature = "headless")]
impl Lightbox<gfx_device_gl::Resources, gfx_device_gl::Factory> {
    pub fn headless_gl(
        width: u16,
        height: u16,
    ) -> Result<
        (
            Self,
            Encoder<gfx_device_gl::Resources, gfx_device_gl::CommandBuffer>,
            gfx_device_gl::Device,
        ),
        Error,
    > {
        use glutin::{Api, GlContext, GlProfile, GlRequest, HeadlessRendererBuilder};

        let context = HeadlessRendererBuilder::new(width as u32, height as u32)
            .with_gl(GlRequest::Specific(Api::OpenGl, (3, 2)))
            .with_gl_profile(GlProfile::Core)
            .build()
            .map_err(|e| failure::err_msg(format!("{}", e)))?;

        unsafe { context.make_current().unwrap() };

        let (device, mut factory) =
            gfx_device_gl::create(|s| context.get_proc_address(s) as *const _);

        let depth_target = factory.create_depth_stencil(width, height)?.2;

        let color_texture = factory.create_texture(
            Kind::D2(width, height, AaMode::Single),
            1,
            Bind::SHADER_RESOURCE | Bind::RENDER_TARGET | Bind::TRANSFER_SRC,
            Usage::Data,
            Some(ChannelType::Srgb),
        )?;
        let color_target = factory.view_texture_as_render_target(&color_texture, 0, None)?;

        let pso_cell = Box::new(
            debug_watcher_pso_cell!(
                pipe = pipe,
                vertex_shader = "shader/vert.glsl",
                fragment_shader = "shader/frag.glsl",
                factory = factory.clone(),
                primitive = Primitive::TriangleList,
                raterizer = Rasterizer::new_fill()
            ).map_err(|e| failure::err_msg(format!("{}", e)))?,
        );

        let encoder = factory.create_command_buffer().into();

        let download_buffer =
            factory.create_download_buffer::<u8>(4 * width as usize * height as usize)?;

        Ok((
            Self {
                width,
                height,
                factory,
                pso_cell,
                download_buffer,
                color_target,
                depth_target,
            },
            encoder,
            device,
        ))
    }
}

impl<R: Resources, F: Factory<R> + Clone + 'static> Lightbox<R, F> {
    pub fn new(width: u16, height: u16, mut factory: F) -> Result<Self, Error> {
        let depth_target = factory.create_depth_stencil(width, height)?.2;

        let color_texture = factory.create_texture(
            Kind::D2(width, height, AaMode::Single),
            1,
            Bind::SHADER_RESOURCE | Bind::RENDER_TARGET | Bind::TRANSFER_SRC,
            Usage::Data,
            Some(ChannelType::Srgb),
        )?;
        let color_target = factory.view_texture_as_render_target(&color_texture, 0, None)?;

        let pso_cell = Box::new(
            debug_watcher_pso_cell!(
                pipe = pipe,
                vertex_shader = "shader/vert.glsl",
                fragment_shader = "shader/frag.glsl",
                factory = factory.clone(),
                primitive = Primitive::TriangleList,
                raterizer = Rasterizer::new_fill()
            ).map_err(|e| failure::err_msg(format!("{}", e)))?,
        );

        let download_buffer =
            factory.create_download_buffer::<u8>(4 * width as usize * height as usize)?;

        Ok(Self {
            width,
            height,
            factory,
            pso_cell,
            download_buffer,
            color_target,
            depth_target,
        })
    }
    pub fn capture<C: CommandBuffer<R>, D: Device<Resources = R, CommandBuffer = C>>(
        &mut self,
        model: &Model<R>,
        model_view_projection: [[f32; 4]; 4],
        encoder: &mut Encoder<R, C>,
        device: &mut D,
    ) -> Result<RgbaImage, Error> {
        let mut data = model.data.clone();
        data.model_view_projection = model_view_projection;

        encoder.clear(&self.color_target, [0.0, 0.0, 0.0, 0.0]);
        encoder.clear_depth(&self.depth_target, 1.0);

        encoder.draw(&model.slice, self.pso_cell.pso(), &data);
        encoder
            .copy_texture_to_buffer_raw(
                model.data.color.raw().get_texture(),
                None,
                gfx::texture::RawImageInfo {
                    xoffset: 0,
                    yoffset: 0,
                    zoffset: 0,
                    width: self.width,
                    height: self.height,
                    depth: 0,
                    format: Srgba8::get_format(),
                    mipmap: 0,
                },
                self.download_buffer.raw(),
                0,
            ).map_err(|e| failure::err_msg(format!("{:?}", e)))?;
        encoder.flush(device);

        let data = self.factory.read_mapping(&self.download_buffer)?.to_vec();
        let image = RgbaImage::from_raw(self.width.into(), self.height.into(), data).unwrap();
        Ok(image::imageops::flip_vertical(&image))
    }
    pub fn capture_billboards<C: CommandBuffer<R>, D: Device<Resources = R, CommandBuffer = C>>(
        &mut self,
        model: &Model<R>,
        encoder: &mut Encoder<R, C>,
        device: &mut D,
    ) -> Result<Vec<RgbaImage>, Error> {
        let [x0, y0, z0] = model.min;
        let [x1, y1, z1] = model.max;

        let [rx, rz] = [x0.abs().max(x1.abs()), z0.abs().max(z1.abs())];
        let [l, b, n] = [-rx, y0.min(0.0), -rz];
        let [r, t, f] = [rx, y1, rz];

        #[rustfmt::skip]
        let forward_mvp_matrix = [
            [2.0/(r-l),      0.0,           0.0,          0.0],
            [0.0,            2.0/(t-b),     0.0,          0.0],
            [0.0,            0.0,          -2.0/(f-n),    0.0],
            [-(r+l)/(r-l),  -(t+b)/(t-b),  -(f+n)/(f-n),  1.0],
        ];

        #[rustfmt::skip]
        let right_mvp_matrix = [
            [0.0,            0.0,          -2.0/(r-l),    0.0],
            [0.0,            2.0/(t-b),     0.0,          0.0],
            [2.0/(f-n),      0.0,           0.0,          0.0],
            [-(f+n)/(f-n),  -(t+b)/(t-b),  -(r+l)/(r-l),  1.0],
        ];

        #[rustfmt::skip]
        let top_mvp_matrix = [
            [2.0/(r-l),      0.0,           0.0,          0.0],
            [0.0,            0.0,           2.0/(t-b),    0.0],
            [0.0,           -2.0/(f-n),     0.0,          0.0],
            [-(r+l)/(r-l),  -(f+n)/(f-n),  -(t+b)/(t-b),  1.0],
        ];

        let forward = self.capture(model, forward_mvp_matrix, encoder, device)?;
        let right = self.capture(model, right_mvp_matrix, encoder, device)?;
        let top = self.capture(model, top_mvp_matrix, encoder, device)?;

        Ok(vec![forward, right, top])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teapot() {
        let (mut lightbox, mut encoder, mut device) = Lightbox::headless_gl(1024, 1024).unwrap();
        let mut model: &[u8] = include_bytes!("../teapot.obj");
        let model = Obj::load_buf(&mut model).unwrap();
        let model = Model::from_obj(&mut lightbox, model, None, HashMap::new()).unwrap();

        let mvp_matrix = [
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let render = lightbox
            .capture(&model, mvp_matrix, &mut encoder, &mut device)
            .unwrap();
        assert_eq!(render.dimensions(), (1024, 1024));
        //render.save("teapot-test.png").unwrap();

        let render = render.into_raw();
        let reference_render = image::load_from_memory(include_bytes!("../teapot.png")).unwrap();

        let mut square_difference = 0;
        for (p, rp) in render
            .into_iter()
            .zip(reference_render.raw_pixels().into_iter())
        {
            let diff = p as i64 - rp as i64;
            square_difference += diff * diff;
        }

        let root_mean_square_difference =
            f64::sqrt(square_difference as f64 / (3.0 * 1024.0 * 1024.0));

        assert!(
            root_mean_square_difference < 2.0,
            "rmsd = {}",
            root_mean_square_difference
        );
    }
}

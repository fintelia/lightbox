#![feature(underscore_imports)]

extern crate failure;
extern crate gfx_core;
extern crate gfx_device_gl;
extern crate glutin;
extern crate image;
extern crate obj;

#[macro_use]
extern crate gfx;
#[macro_use]
extern crate gfx_shader_watch;

use gfx::{CommandBuffer, Device, Encoder, Factory, Factory as GfxFactory, IndexBuffer, Primitive,
          Resources, Slice};
use gfx::traits::FactoryExt;
use gfx::state::Rasterizer;
use gfx::texture::{AaMode, Kind};
use gfx::memory::{Bind, Usage};
use gfx_core::memory::Typed;
use gfx_core::format::{ChannelType, DepthStencil, Formatted, Srgba8};
use gfx_core::handle::{Buffer, DepthStencilView, RenderTargetView};
use gfx_shader_watch::PsoCell;
use image::RgbaImage;
use obj::{Obj, SimplePolygon};
use glutin::{Api, GlContext, GlProfile, GlRequest, HeadlessRendererBuilder};
use failure::Error;

gfx_defines!{
    vertex Vertex {
        position: [f32; 3] = "vPosition",
        texcoord: [f32; 2] = "vTexcoord",
        normal: [f32; 3] = "vNormal",
    }
}
gfx_pipeline!( pipe {
    vertices: gfx::VertexBuffer<Vertex> = (),

    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",

    color: gfx::RenderTarget<Srgba8> = "OutColor",
    depth: gfx::DepthTarget<DepthStencil> = gfx::preset::depth::PASS_WRITE,
});

pub struct Model<R: Resources> {
    slice: Slice<R>,
    data: pipe::Data<R>,
}
impl<R: Resources> Model<R> {
    pub fn from_obj<'a, F, C, D>(
        lightbox: &mut Lightbox<R, F, C, D>,
        mut model: Obj<'a, SimplePolygon>,
    ) -> Self
    where
        F: Factory<R>,
        C: CommandBuffer<R>,
        D: Device<Resources = R, CommandBuffer = C>,
    {
        if model.normal.is_empty() {
            model.normal.push([0.0, 0.0, 0.0]);
        }
        if model.texture.is_empty() {
            model.texture.push([0.0, 0.0]);
        }

        let mut vertices = Vec::new();
        for object in &model.objects {
            for group in &object.groups {
                // TODO: use material
                let _material = &group.material;

                for poly in &group.polys {
                    for i in 0..poly.len().checked_sub(2).unwrap() {
                        for j in 0..3 {
                            vertices.push(Vertex {
                                position: model.position[poly[i + j].0],
                                texcoord: model.texture[poly[i + j].1.unwrap_or(0)],
                                normal: model.normal[poly[i + j].2.unwrap_or(0)],
                            });
                        }
                    }
                }
            }
        }

        let (vertices, slice) = lightbox
            .factory
            .create_vertex_buffer_with_slice(&vertices, IndexBuffer::Auto);

        Self {
            slice,
            data: pipe::Data {
                color: lightbox.color_target.clone(),
                depth: lightbox.depth_target.clone(),
                model_view_projection: [[-12.0; 4]; 4],
                vertices,
            },
        }
    }
}

pub struct Lightbox<R, F, C, D>
where
    R: Resources,
    F: Factory<R>,
    C: CommandBuffer<R>,
    D: Device<Resources = R, CommandBuffer = C>,
{
    width: u16,
    height: u16,

    device: D,
    factory: F,

    encoder: Encoder<R, C>,

    download_buffer: Buffer<R, u8>,

    pso_cell: Box<PsoCell<R, F, pipe::Init<'static>>>,
    color_target: RenderTargetView<R, Srgba8>,
    depth_target: DepthStencilView<R, DepthStencil>,
}
impl
    Lightbox<
        gfx_device_gl::Resources,
        gfx_device_gl::Factory,
        gfx_device_gl::CommandBuffer,
        gfx_device_gl::Device,
    >
{
    pub fn headless_gl(width: u16, height: u16) -> Result<Self, Error> {
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

        let pso_cell = Box::new(debug_watcher_pso_cell!(
            pipe = pipe,
            vertex_shader = "shader/vert.glsl",
            fragment_shader = "shader/frag.glsl",
            factory = factory.clone(),
            primitive = Primitive::TriangleList,
            raterizer = Rasterizer::new_fill()
        ).map_err(|e| failure::err_msg(format!("{}", e)))?);

        let encoder = factory.create_command_buffer().into();

        let download_buffer =
            factory.create_download_buffer::<u8>(4 * width as usize * height as usize)?;

        Ok(Self {
            width,
            height,
            device,
            factory,
            encoder,
            pso_cell,
            download_buffer,
            color_target,
            depth_target,
        })
    }
}

impl<R, F, C, D> Lightbox<R, F, C, D>
where
    R: Resources,
    F: Factory<R> + Clone + 'static,
    C: CommandBuffer<R>,
    D: Device<Resources = R, CommandBuffer = C>,
{
    pub fn new(
        width: u16,
        height: u16,
        device: D,
        mut factory: F,
        encoder: Encoder<R, C>,
    ) -> Result<Self, Error> {
        let depth_target = factory.create_depth_stencil(width, height)?.2;

        let color_texture = factory.create_texture(
            Kind::D2(width, height, AaMode::Single),
            1,
            Bind::SHADER_RESOURCE | Bind::RENDER_TARGET | Bind::TRANSFER_SRC,
            Usage::Data,
            Some(ChannelType::Srgb),
        )?;
        let color_target = factory.view_texture_as_render_target(&color_texture, 0, None)?;

        let pso_cell = Box::new(debug_watcher_pso_cell!(
            pipe = pipe,
            vertex_shader = "shader/vert.glsl",
            fragment_shader = "shader/frag.glsl",
            factory = factory.clone(),
            primitive = Primitive::TriangleList,
            raterizer = Rasterizer::new_fill()
        ).map_err(|e| failure::err_msg(format!("{}", e)))?);

        let download_buffer =
            factory.create_download_buffer::<u8>(4 * width as usize * height as usize)?;

        Ok(Self {
            width,
            height,
            device,
            factory,
            encoder,
            pso_cell,
            download_buffer,
            color_target,
            depth_target,
        })
    }
    pub fn capture(
        &mut self,
        model: &Model<R>,
        model_view_projection: [[f32; 4]; 4],
    ) -> Result<RgbaImage, Error> {
        let mut data = model.data.clone();
        data.model_view_projection = model_view_projection;

        self.encoder.clear(&self.color_target, [0.0, 1.0, 0.0, 1.0]);
        self.encoder.clear_depth(&self.depth_target, 1.0);

        self.encoder.draw(&model.slice, self.pso_cell.pso(), &data);
        self.encoder
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
            )
            .map_err(|e| failure::err_msg(format!("{:?}", e)))?;
        self.encoder.flush(&mut self.device);

        let data = self.factory.read_mapping(&self.download_buffer)?.to_vec();
        let image = RgbaImage::from_raw(self.width.into(), self.height.into(), data).unwrap();
        Ok(image::imageops::flip_vertical(&image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teapot() {
        let mut lightbox = Lightbox::headless_gl(1024, 1024).unwrap();
        let mut model: &[u8] = include_bytes!("../teapot.obj");
        let model = Obj::load_buf(&mut model).unwrap();
        let model = Model::from_obj(&mut lightbox, model);

        let mvp_matrix = [
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let render = lightbox.capture(&model, mvp_matrix).unwrap();
        assert_eq!(render.dimensions(), (1024, 1024));
        // render.save("teapot-test.png")?;

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

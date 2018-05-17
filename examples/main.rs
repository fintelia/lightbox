extern crate camera_controllers;
extern crate lightbox;
extern crate obj;

use camera_controllers::Camera;
use lightbox::{Lightbox, Model};
use obj::Obj;

fn main() {
    let (mut lightbox, mut encoder, mut device) = Lightbox::headless_gl(1024, 1024).unwrap();
    let mut model: &[u8] = include_bytes!("../teapot.obj");
    let model = Obj::load_buf(&mut model).unwrap();
    let model = Model::from_obj(&mut lightbox, model);

    let camera = Camera::new([0.0, 0.0, 0.0]);

    lightbox
        .capture(&model, camera.orthogonal(), &mut encoder, &mut device)
        .unwrap()
        .save("teapot.png")
        .unwrap();
}

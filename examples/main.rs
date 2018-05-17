extern crate camera_controllers;
extern crate lightbox;
extern crate obj;

use camera_controllers::Camera;
use obj::Obj;
use lightbox::{Lightbox, Model};

fn main() {
    let mut lightbox = Lightbox::headless_gl(1024, 1024).unwrap();
    let mut model: &[u8] = include_bytes!("../teapot.obj");
    let model = Obj::load_buf(&mut model).unwrap();
    let model = Model::from_obj(&mut lightbox, model);

    let camera = Camera::new([0.0, 0.0, 0.0]);

    lightbox
        .capture(&model, camera.orthogonal())
        .unwrap()
        .save("teapot.png")
        .unwrap();
}

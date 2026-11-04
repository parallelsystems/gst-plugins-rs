// SPDX-License-Identifier: MPL-2.0

use gst::glib;
use gst::prelude::*;
mod imp;

glib::wrapper! {
    pub struct Pacer(ObjectSubclass<imp::Pacer>) @extends gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "rtppacer",
        gst::Rank::NONE,
        Pacer::static_type(),
    )
}

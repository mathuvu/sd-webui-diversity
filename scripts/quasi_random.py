import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks, devices
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from modules.rng import ImageRNG

import torch
from nevergrad.common import sphere

class QuasiRandomRNG(ImageRNG):
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, method="none"):
             super(QuasiRandomRNG, self).__init__(shape, seeds, subseeds, subseed_strength, seed_resize_from_h, seed_resize_from_w)
             self.method = method
    
    def next(self):
            xs = super().next()
            xs =  torch.FloatTensor(sphere.quasi_randomize(xs.cpu().numpy(), self.method)).to(device=devices.device)
            return xs
    
    @classmethod
    def init_from_ImageRNG(cls, im_rng: ImageRNG, method='none'):
            return cls(im_rng.shape, im_rng.seeds, im_rng.subseeds, im_rng.subseed_strength, im_rng.seed_resize_from_h, im_rng.seed_resize_from_w, method=method)
    

class Script(scripts.Script):
        # Extension title in menu UI
        def title(self):
                return "Quasi-random init"

        def show(self, is_img2img):
                return scripts.AlwaysVisible
        
        def process_batch(self, p, *args, **kwargs):
                if args[1] and len(kwargs['seeds']) > 1:
                        p.rng = QuasiRandomRNG.init_from_ImageRNG(p.rng, method=args[1])
                return 

        # Setup menu ui detail
        def ui(self, is_img2img):
                choices = ["none"]
                with gr.Accordion('Quasi-random init', open=False):
                        with gr.Row():
                                use_quasi_random = gr.Checkbox(
                                        False,
                                        label="enable"
                                )
                                method_name = gr.Dropdown(label='method', elem_id=f"quasi_random_method", choices=choices, value=choices[0])
                return [use_quasi_random, method_name]

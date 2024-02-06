import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks, devices
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from modules.rng import ImageRNG

import torch
import numpy as np
from nevergrad.common import sphere

class QuasiRandomRNG(ImageRNG):
        def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, method="none"):
                super(QuasiRandomRNG, self).__init__(shape, seeds, subseeds, subseed_strength, seed_resize_from_h, seed_resize_from_w)
                self.method = method

        def next(self):
                xs = super().next()
                xs = np.moveaxis(xs.cpu().numpy(), 1, 3)
                xs = sphere.quasi_randomize(xs, self.method)
                xs = np.moveaxis(xs, 3, 1)
                xs =  torch.FloatTensor(xs).to(device=devices.device)
                return xs

        @classmethod
        def init_from_ImageRNG(cls, im_rng: ImageRNG, method='none'):
                return cls(im_rng.shape, im_rng.seeds, im_rng.subseeds, im_rng.subseed_strength, im_rng.seed_resize_from_h, im_rng.seed_resize_from_w, method=method)
    
class DummyImageRNG:
        def __init__(self, full_lattent_batch, batch_size, batch_number=0):
                self.full_lattent_batch = full_lattent_batch
                self.batch_size = batch_size
                self.batch_number = batch_number
        
        def first(self):
                if self.batch_number != 0:
                        self.batch_number = 0
                xs = self.full_lattent_batch[:self.batch_size]
                xs = torch.FloatTensor(xs).to(device=devices.device)
                return xs
        
        def next(self):
                xs = self.full_lattent_batch[self.batch_number * self.batch_size:(self.batch_number + 1) * self.batch_size]
                xs = torch.FloatTensor(xs).to(device=devices.device)
                self.batch_number += 1
                return xs

class Script(scripts.Script):
        def __init__(self):
                super().__init__()
                self.FULL_BATCH_LATENT = []

        # Extension title in menu UI
        def title(self):
                return "Quasi-random init"

        def show(self, is_img2img):
                return scripts.AlwaysVisible
        
        def process_batch(self, p, *args, **kwargs):
                if not args[0]:
                        return
                if len(kwargs['seeds']) <= 1:
                        return
                if kwargs['batch_number'] == 0 and args[2]:
                        rng = ImageRNG(p.rng.shape, p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
                        self.FULL_BATCH_LATENT = rng.next()
                        self.FULL_BATCH_LATENT = self.FULL_BATCH_LATENT.cpu().numpy()
                        self.FULL_BATCH_LATENT = np.moveaxis(self.FULL_BATCH_LATENT, 1, 3)
                        self.FULL_BATCH_LATENT = sphere.quasi_randomize(self.FULL_BATCH_LATENT, args[1])
                        self.FULL_BATCH_LATENT = np.moveaxis(self.FULL_BATCH_LATENT, 3, 1)
                if args[2]:
                        p.rng = DummyImageRNG(self.FULL_BATCH_LATENT, p.batch_size, batch_number=kwargs['batch_number'])
                else:
                        p.rng = QuasiRandomRNG.init_from_ImageRNG(p.rng, method=args[1])
                return 

        # Setup menu ui detail
        def ui(self, is_img2img):
                choices = ["none"]
                with gr.Accordion('Quasi-random init', open=True):
                        with gr.Row():
                                use_quasi_random = gr.Checkbox(
                                        True,
                                        label="enable"
                                )
                                method_name = gr.Dropdown(label='method', elem_id=f"quasi_random_method", choices=choices, value=choices[0])
                                use_batch_count = gr.Checkbox(
                                        False,
                                        label="full batch"
                                )
                return [use_quasi_random, method_name, use_batch_count]

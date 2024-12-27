# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from .camera.attention import TemporalPoseCondTransformerBlock as TemporalBasicTransformerBlock
from diffusers.models.attention import BasicTransformerBlock
from torch import nn

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int, lora_scale: Optional[float] = None
):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    if lora_scale is None:
        ff_output = torch.cat(
            [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
            dim=chunk_dim,
        )
    else:
        # TOOD(Patrick): LoRA scale can be removed once PEFT refactor is complete
        ff_output = torch.cat(
            [ff(hid_slice, scale=lora_scale) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
            dim=chunk_dim,
        )

    return ff_output


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
            self_attention_additional_feats=None,
            mode=None,
        ):
            batch_size = hidden_states.shape[0]

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Retrieve lora scale.
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            # 2. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    # print("this is write")
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

                if MODE == "read":
                    # bank_fea = [
                    #     rearrange(
                    #         d.unsqueeze(1).repeat(1, video_length, 1, 1),
                    #         "b t l c -> (b t) l c",
                    #     )
                    #     for d in self.bank
                    # ]
                    bank_fea=[]
                    for d in self.bank:
                        if d.shape[0]==1:
                            bank_fea.append(d.repeat(norm_hidden_states.shape[0],1,1))
                        else:
                            bank_fea.append(d)

                    modify_norm_hidden_states = torch.cat(
                        [norm_hidden_states] + bank_fea, dim=1
                    )
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=modify_norm_hidden_states,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    if self.use_ada_layer_norm_zero:
                        attn_output = gate_msa.unsqueeze(1) * attn_output
                    elif self.use_ada_layer_norm_single:
                        attn_output = gate_msa * attn_output

                    hidden_states = attn_output + hidden_states
                    if hidden_states.ndim == 4:
                        hidden_states = hidden_states.squeeze(1)

                    # 2.5 GLIGEN Control
                    if gligen_kwargs is not None:
                        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

                    # 3. Cross-Attention
                    if self.attn2 is not None:
                        if self.use_ada_layer_norm:
                            norm_hidden_states = self.norm2(hidden_states, timestep)
                        elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                            norm_hidden_states = self.norm2(hidden_states)
                        elif self.use_ada_layer_norm_single:
                            # For PixArt norm2 isn't applied here:
                            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                            norm_hidden_states = hidden_states
                        else:
                            raise ValueError("Incorrect norm")

                        if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                            norm_hidden_states = self.pos_embed(norm_hidden_states)

                        attn_output = self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=encoder_attention_mask,
                            **cross_attention_kwargs,
                        )
                        hidden_states = attn_output + hidden_states

                    # 4. Feed-forward
                    if not self.use_ada_layer_norm_single:
                        norm_hidden_states = self.norm3(hidden_states)

                    if self.use_ada_layer_norm_zero:
                        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                    if self.use_ada_layer_norm_single:
                        norm_hidden_states = self.norm2(hidden_states)
                        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                    if self._chunk_size is not None:
                        # "feed_forward_chunk_size" can be used to save memory
                        ff_output = _chunked_feed_forward(
                            self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
                        )
                    else:
                        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

                    if self.use_ada_layer_norm_zero:
                        ff_output = gate_mlp.unsqueeze(1) * ff_output
                    elif self.use_ada_layer_norm_single:
                        ff_output = gate_mlp * ff_output

                    hidden_states = ff_output + hidden_states
                    if hidden_states.ndim == 4:
                        hidden_states = hidden_states.squeeze(1)

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
    
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.use_ada_layer_norm_single:
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if not self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    # or isinstance(module, TemporalBasicTransformerBlock) 
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    # or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                # if isinstance(module, TemporalBasicTransformerBlock):             
                #     module.forward = hacked_basic_transformer_inner_forward.__get__(
                #         module, TemporalBasicTransformerBlock
                #     )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:


            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                # reader_attn_modules = [
                #     module
                #     for module in torch_dfs(self.unet)
                #     if isinstance(module, TemporalBasicTransformerBlock)
                # ]
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]                
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]
                # w.bank.clear()

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    # or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    # or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()

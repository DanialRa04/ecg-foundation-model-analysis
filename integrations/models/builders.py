"""Original loading glue for locally installed author implementations."""
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

_ACTIVE_SOURCE = None

def prepare(context, *subdirectories):
    global _ACTIVE_SOURCE
    source = Path(context['source']).resolve()
    if _ACTIVE_SOURCE is not None and _ACTIVE_SOURCE != source:
        raise RuntimeError('Use a separate extraction process for each author implementation')
    if not source.is_dir(): raise FileNotFoundError('Local author implementation directory is missing')
    paths = [*context.get('dependencies', []), *[source / sub for sub in subdirectories], source]
    for path in paths:
        if not Path(path).is_dir(): raise FileNotFoundError(f'Local dependency directory is missing: {path}')
    sys.path[:0] = [str(path) for path in paths]
    _ACTIVE_SOURCE = source

def checkpoint(context):
    path = Path(context['checkpoint'])
    if not path.is_file(): raise FileNotFoundError('Checkpoint file is missing')
    if not context.get('trust_checkpoint', False):
        raise ValueError('Pickle-based author weights require --trust-checkpoint; use locally trusted files')
    return torch.load(path, map_location='cpu', mmap=True, weights_only=not context.get('trust_checkpoint', False))

def state(model, values, allow_extra_prefixes=()):
    missing, unexpected = model.load_state_dict(values, strict=False)
    unexpected = [key for key in unexpected if not any(key.startswith(prefix) for prefix in allow_extra_prefixes)]
    if missing or unexpected:
        raise ValueError(f'Checkpoint mismatch: {len(missing)} missing and {len(unexpected)} unexpected keys')

def finish(model, context, forward, pooling):
    model.requires_grad_(False)
    model.to(context['device']).eval()
    return model, forward, {'architecture': f'{type(model).__module__}.{type(model).__name__}', 'pooling': pooling}

def ecgfounder(context):
    prepare(context)
    from net1d import Net1D
    model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64,160,160,400,400,1024,1024], m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16, stride=2, groups_width=16, n_classes=150, use_bn=False, use_do=False, return_features=True)
    state(model, checkpoint(context)['state_dict'])
    return finish(model, context, lambda batch: model(batch)[1], 'official deep_features')

def merl(context):
    prepare(context, 'utils')
    from vit1d import vit_tiny
    model = vit_tiny(num_leads=12)
    state(model, checkpoint(context))
    return finish(model, context, model.forward_encoding, 'official forward_encoding patch mean and LayerNorm')

def ecgflow(context):
    prepare(context)
    import ecgflow.models.mtsm
    from timm import create_model
    from safetensors.torch import load_file
    model = create_model('mtsm_base_patch50', img_size=5000, in_chans=8)
    values = load_file(str(context['checkpoint']), device='cpu')
    values = {key.removeprefix('model.'): value for key, value in values.items()}
    # Decoder weights belong to the released pretraining model as well.
    state(model, values)
    return finish(model, context, lambda batch: model.forward_encoder(batch, mask_ratio=0.0)[0][:, 1:].mean(1), 'mean normalized patch tokens, excluding CLS')

def ecgfm_ked(context):
    prepare(context)
    from models.xresnet1d_101 import xresnet1d101
    model = xresnet1d101(num_classes=105, input_channels=12, kernel_size=5, ps_head=.5, use_ecgNet_Diagnosis='other')
    state(model, checkpoint(context)['ecg_model'])
    def forward(batch):
        value = model(batch)
        return value.mean(-1) if value.ndim == 3 else value
    return finish(model, context, forward, 'ECG encoder output, temporal mean when rank three')

def clef(context):
    prepare(context)
    from clef.baselines.models.CLEF import create_net1d_by_size
    model = create_net1d_by_size(device=context['device'], model_size='medium', pth=None, in_channels=1)
    data = checkpoint(context)
    values = data.get('state_dict', data)
    values = {key.removeprefix('backbone.'): value for key, value in values.items()}
    state(model, values)
    def forward(batch):
        value = model(batch)
        return value.mean(-1) if value.ndim == 3 else value
    return finish(model, context, forward, 'pooled backbone features')

def ecg_jepa(context):
    prepare(context)
    from ecg_jepa import MaskTransformer
    original_linspace = torch.linspace
    def cpu_schedule(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_linspace(*args, **kwargs)
    try:
        # Drop-path schedules are scalar constructor settings, not parameters.
        torch.linspace = cpu_schedule
        with torch.device('meta'):
            model = MaskTransformer(embed_dim=768, depth=12, num_heads=16, c=8, p=50, t=50, pos_type='sincos', mask_scale=(0,0), leads=list(range(8)))
    finally:
        torch.linspace = original_linspace
    torch.nn.Module.load_state_dict(model, checkpoint(context)['encoder'], strict=True, assign=True)
    return finish(model, context, model.representation, 'official encoder representation')

def st_mem(context):
    prepare(context)
    from models.encoder.st_mem_vit import st_mem_vit_base
    model = st_mem_vit_base(num_leads=12, seq_len=2250, patch_size=75)
    state(model, checkpoint(context)['model'])
    return finish(model, context, model.forward_encoding, 'official separator removal, lead/patch mean and LayerNorm')

def hubert_ecg(context):
    prepare(context)
    import hubert_ecg
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(context['checkpoint']), local_files_only=True, trust_remote_code=False, use_safetensors=True)
    return finish(model, context, lambda batch: model(batch.flatten(1), return_dict=True).last_hidden_state.mean(1), 'mean last hidden states of flattened leads')

def ecg_fm(context):
    return _streamed_fairseq(context, large=False)

def deepecg_ssl(context):
    return _streamed_fairseq(context, large=False)

def ecg_lfm(context):
    return _streamed_fairseq(context, large=True)

def _streamed_fairseq(context, large):
    prepare(context)
    from fairseq_signals.models import build_model
    from omegaconf import OmegaConf
    from common.checkpoints import open_structure, read_tensor
    archive, prefix, config, descriptors = open_structure(context['checkpoint'],context.get('trust_checkpoint',False))
    cfg = config if OmegaConf.is_config(config) else OmegaConf.create(config)
    modules = [importlib.import_module('fairseq_signals.modules.transformer_encoder')]
    if large: modules.append(importlib.import_module('fairseq_signals.models.ECGLFM.ECGLFM'))
    originals = [module.init_bert_params for module in modules]
    try:
        for module in modules: module.init_bert_params = lambda _module: None
        with torch.device('meta'):
            model = build_model(cfg.model, task=None, from_checkpoint=large)
    finally:
        for module, original in zip(modules, originals): module.init_bert_params = original
    dtype = torch.float16 if large and str(context['device']).startswith('cuda') else torch.float32
    values = {}
    try:
        for index,(key,descriptor) in enumerate(descriptors.items(),1):
            try:
                value = read_tensor(archive,prefix,descriptor)
                values[key] = value.to(device=context['device'],dtype=dtype if value.is_floating_point() else value.dtype)
            except (MemoryError, torch.OutOfMemoryError) as error:
                raise RuntimeError(f'Insufficient memory loading tensor {key}, shape={descriptor["size"]}') from error
            del value
            if index % 100 == 0: print(f'Loaded {index}/{len(descriptors)} model tensors',flush=True)
        torch.nn.Module.load_state_dict(model, values, strict=True, assign=True)
    finally: archive.close()
    positions = getattr(getattr(model,'encoder',None), 'embed_positions', None)
    if positions is not None and getattr(positions, 'pe', None) is not None and positions.pe.device.type == 'meta': positions.pe = None
    def forward(batch):
        padding = torch.zeros((len(batch), batch.shape[-1]), dtype=torch.bool, device=batch.device)
        return model.extract_features(batch.to(dtype=dtype), padding_mask=padding, mask=False)['x'].mean(1)
    result = finish(model, context, forward, 'mean sequence tokens; float16 for large CUDA encoder, otherwise float32')
    result[2]['checkpoint_loading'] = 'streamed model tensors; optimizer storage omitted'
    return result

def _cauchy(v, z, w):
    v, w = v.unsqueeze(-1), w.unsqueeze(-1)
    z = z.reshape((1,) * (v.ndim - 1) + (z.numel(),))
    return (v / (z - w) + v.conj() / (z - w.conj())).sum(-2)

def _s4(context, cpc, expected_normalize):
    code = 'code' if cpc else 'downstream_evaluation/code'
    prepare(context, code)
    from hydra import compose, initialize_config_dir
    from clinical_ts.config import create_default_config
    from clinical_ts.template_modules import ShapeConfig, _string_to_class
    import clinical_ts.ts.s4_modules.s42 as s42
    s42.cauchy_conj = _cauchy
    config_path = Path(context['config']).resolve()
    if not config_path.is_file(): raise FileNotFoundError('Matching model YAML is missing')
    create_default_config()
    search = (Path(context['source']) / code / 'clinical_ts/models/conf').as_posix()
    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        cfg = compose(config_name=config_path.stem, overrides=[f'hydra.searchpath=[{json.dumps(search)}]'])
    if bool(cfg.base.normalize) != expected_normalize:
        raise ValueError('Model configuration normalization differs from its preprocessing recipe')
    length = 2400 if cpc else 600
    model = _string_to_class(cfg.ts._target_)(cfg.ts, ShapeConfig(channels=12, length=length, sequence_last=True), None)
    values = {key.removeprefix('ts_encoder.'): value for key, value in checkpoint(context)['state_dict'].items() if key.startswith('ts_encoder.')}
    missing = set(model.state_dict()) - set(values)
    if missing: raise ValueError(f'S4 checkpoint is missing {len(missing)} encoder keys')
    for name, parameter in model.named_parameters(): parameter.data = values[name].data
    for name, buffer in model.named_buffers(): buffer.data = values[name].data
    return finish(model, context, lambda batch: model(seq=batch)['seq'].mean(1), 'mean final TimeSeriesEncoder sequence')

def s4(context, variant):
    result = _s4(context, cpc=False, expected_normalize=variant != 'cpc')
    result[2]['variant'] = variant
    return result

def ecg_cpc(context):
    return _s4(context, cpc=True, expected_normalize=False)

def heartlang(context):
    prepare(context)
    from modeling_pretrain import HeartLang
    from QRSTokenizer import QRSTokenizer
    model = HeartLang(pretrained=False)
    state(model, checkpoint(context)['model'])
    def forward(batch):
        tokenizer = QRSTokenizer(fs=100, max_len=256, token_len=96, save_path=str(context['output']), stage='inference', used_channels=list(range(12)))
        sequences, channels, times = [], [], []
        for signal in batch.detach().cpu().numpy():
            qrs = tokenizer.qrs_detection(signal)
            segments = tokenizer.extract_qrs_segments(signal, qrs)
            sequence, channel, time = tokenizer.qrs_to_sequence(segments, qrs)
            sequences.append(sequence); channels.append(channel); times.append(time)
        tokens = torch.as_tensor(np.asarray(sequences), dtype=torch.float32, device=batch.device)
        channels = torch.as_tensor(np.asarray(channels), dtype=torch.long, device=batch.device)
        times = torch.as_tensor(np.asarray(times), dtype=torch.long, device=batch.device)
        return model.forward_feature(tokens, in_chan_matrix=channels, in_time_matrix=times)[:,0]
    return finish(model, context, forward, 'QRS tokens; final post-normalization CLS')

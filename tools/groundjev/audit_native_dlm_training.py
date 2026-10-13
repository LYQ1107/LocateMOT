"""Meta-device reconstruction of the official DLM training wrapper and weight contract."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/GroundAnything')]


def main():
    import torch
    from transformers import AutoConfig,AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors import safe_open
    from models.dlm.vlm import GAMQwen3DLM
    checkpoint=ROOT/'models/GroundAnything-DLM'
    raw=AutoConfig.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
    Config=get_class_from_dynamic_module('configuration_groundinganything.GroundAnythingVLMConfig',str(checkpoint),local_files_only=True)
    Model=get_class_from_dynamic_module('modeling_groundinganything.GroundAnythingVLMForConditionalGeneration',str(checkpoint),local_files_only=True)
    config=Config.from_dict(raw.to_dict());config._attn_implementation='eager'
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
    mask=tokenizer.encode('|<MASK>|',add_special_tokens=False)
    assert len(mask)==1
    with safe_open(checkpoint/'model.safetensors',framework='pt') as source:
        actual={name:source.get_slice(name).get_shape() for name in source.keys()}
    vocabulary_rows=actual['base_model.model.language_model.embed_tokens.weight'][0]
    assert vocabulary_rows==len(tokenizer) and mask[0]<vocabulary_rows
    original_vocabulary_rows=config.text_config.vocab_size
    # The released wrapper includes the native added MASK row; its AR-shaped
    # JSON retains the pre-conversion count. Reconstruct exactly the real
    # checkpoint shape, without resizing or synthesizing any weight values.
    config.text_config.vocab_size=vocabulary_rows
    with torch.device('meta'):
        base=Model(config)
        wrapper=GAMQwen3DLM(base,mask_token_id=mask[0],im_end_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),block_size=32)
    expected={name:list(tensor.shape) for name,tensor in wrapper.state_dict().items()}
    missing=sorted(set(expected)-set(actual));unexpected=sorted(set(actual)-set(expected))
    mismatched={name:{'expected':shape,'checkpoint':actual[name]} for name,shape in expected.items() if name in actual and shape!=actual[name]}
    report={'status':'EXACT_NATIVE_DLM_TRAINING_WRAPPER_KEYS_VALIDATED' if not (missing or unexpected or mismatched) else 'NATIVE_DLM_WRAPPER_KEY_MISMATCH',
        'checkpoint':str(checkpoint),'native_wrapper':'official models.dlm.vlm.GAMQwen3DLM',
        'base_model_class':str(type(base)),'metadata_reconstruction_only':True,'tensors':len(actual),
        'missing':missing,'unexpected':unexpected,'shape_mismatches':mismatched,
        'release_vocabulary_contract':{'JSON_rows':original_vocabulary_rows,'actual_checkpoint_rows':vocabulary_rows,'tokenizer_rows':len(tokenizer),'MASK_id':mask[0],'new_weight_values_generated':False},
        'native_loss_contract':{'mdm_loss_weight':wrapper.mdm_loss_weight,'causal_loss_weight':wrapper.causal_loss_weight,'block_size':wrapper.block_size},
        'gradient_training_performed':False,'matching_AR_checkpoint_download_required':False if not (missing or unexpected or mismatched) else 'not determined by a key audit',
        'HF_plain_CausalLM_SFT_used':False,'image_GT_read':False,'language_weights_randomly_substituted':False}
    out=ROOT/'outputs/groundjev/ground_adaptation/native_dlm_training_key_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if missing or unexpected or mismatched:raise RuntimeError('Native training wrapper does not match the real released DLM weights')


if __name__=='__main__':main()

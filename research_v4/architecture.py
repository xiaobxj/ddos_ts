"""Official Crossformer modules with explicit padding masks and a return readout.

The upstream encoder/decoder and parameter layout are retained. The attention
functions below add only key masking and suppression of padded query states.
An explicit per-patch duration/end embedding makes resampling auditable.
"""
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'research_v3/vendor_py'))
sys.path.insert(0,str(ROOT/'vendor/Crossformer'))
import torch
from torch import nn
from cross_models.cross_former import Crossformer


def attention(layer,query,key,value,key_valid=None):
    batch,length,_=query.shape
    count=key.shape[1];heads=layer.n_heads
    q=layer.query_projection(query).view(batch,length,heads,-1)
    k=layer.key_projection(key).view(batch,count,heads,-1)
    v=layer.value_projection(value).view(batch,count,heads,-1)
    scale=layer.inner_attention.scale or 1./math.sqrt(q.shape[-1])
    score=torch.einsum('blhe,bshe->bhls',q,k)*scale
    if key_valid is not None:
        if not bool(key_valid.any(dim=-1).all()):
            raise ValueError('Attention requires at least one valid historical key')
        score=score.masked_fill(~key_valid[:,None,None,:],-torch.inf)
    weights=layer.inner_attention.dropout(torch.softmax(score,dim=-1))
    output=torch.einsum('bhls,bshd->blhd',weights,v).contiguous().view(batch,length,-1)
    return layer.out_projection(output)


def tsa(layer,x,valid):
    batch,dims,segments,width=x.shape
    mask=valid[:,None,:,None]
    x=x.masked_fill(~mask,0.)
    temporal=x.reshape(batch*dims,segments,width)
    keys=valid[:,None,:].expand(batch,dims,segments).reshape(batch*dims,segments)
    encoded=attention(layer.time_attention,temporal,temporal,temporal,keys)
    dim_in=layer.norm1(temporal+layer.dropout(encoded))
    dim_in=layer.norm2(dim_in+layer.dropout(layer.MLP1(dim_in)))
    dim_in=dim_in.reshape(batch,dims,segments,width).masked_fill(~mask,0.)
    dim_send=dim_in.permute(0,2,1,3).reshape(batch*segments,dims,width)
    routers=layer.router[None].expand(batch,-1,-1,-1).reshape(batch*segments,layer.router.shape[1],width)
    buffer=attention(layer.dim_sender,routers,dim_send,dim_send)
    received=attention(layer.dim_receiver,dim_send,buffer,buffer)
    dim_enc=layer.norm3(dim_send+layer.dropout(received))
    dim_enc=layer.norm4(dim_enc+layer.dropout(layer.MLP2(dim_enc)))
    result=dim_enc.reshape(batch,segments,dims,width).permute(0,2,1,3)
    return result.masked_fill(~mask,0.)


def merge(layer,x,valid):
    x=x.masked_fill(~valid[:,None,:,None],0.)
    remainder=x.shape[2]%layer.win_size
    if remainder:
        count=layer.win_size-remainder
        # Retain upstream last-segment repetition, including its validity.
        x=torch.cat([x,x[:,:,-count:,:]],dim=2)
        valid=torch.cat([valid,valid[:,-count:]],dim=1)
    merged=torch.cat([x[:,:,i::layer.win_size,:] for i in range(layer.win_size)],dim=-1)
    next_valid=torch.stack([valid[:,i::layer.win_size] for i in range(layer.win_size)]).any(dim=0)
    merged=layer.linear_trans(layer.norm(merged))
    return merged.masked_fill(~next_valid[:,None,:,None],0.),next_valid


def encode(encoder,x,valid):
    outputs=[x];masks=[valid]
    for block in encoder.encode_blocks:
        if block.merge_layer is not None:
            x,valid=merge(block.merge_layer,x,valid)
        for layer in block.encode_layers:
            x=tsa(layer,x,valid)
        outputs.append(x);masks.append(valid)
    return outputs,masks


def decode(decoder,x,encoded,masks):
    batch,dims,out_segments,width=x.shape
    output_valid=torch.ones((batch,out_segments),dtype=torch.bool,device=x.device)
    result=None
    for layer,cross,valid in zip(decoder.decode_layers,encoded,masks):
        x=tsa(layer.self_attention,x,output_valid)
        flat=x.reshape(batch*dims,out_segments,width)
        other=cross.reshape(batch*dims,cross.shape[2],width)
        keys=valid[:,None,:].expand(batch,dims,valid.shape[1]).reshape(batch*dims,valid.shape[1])
        mixed=attention(layer.cross_attention,flat,other,other,keys)
        flat=layer.norm1(flat+layer.dropout(mixed))
        x=layer.norm2(flat+layer.MLP1(flat)).reshape(batch,dims,out_segments,width)
        prediction=layer.linear_pred(x).reshape(batch,dims*out_segments,-1)
        result=prediction if result is None else result+prediction
    return result.reshape(batch,dims,out_segments,-1).permute(0,2,3,1).reshape(batch,-1,dims)


class SegmentedCrossformer(nn.Module):
    def __init__(self,dims,d_model=32,d_ff=64,n_heads=4,e_layers=2,factor=4,dropout=.1):
        super().__init__()
        self.backbone=Crossformer(data_dim=dims,in_len=125,out_len=5,seg_len=5,win_size=2,
                                  factor=factor,d_model=d_model,d_ff=d_ff,n_heads=n_heads,
                                  e_layers=e_layers,dropout=dropout,baseline=False,device=torch.device('cpu'))
        self.geometry=nn.Linear(2,d_model)
        self.return_head=nn.Linear(5*dims,1)
        nn.init.zeros_(self.return_head.weight)
        nn.init.zeros_(self.return_head.bias)

    def decoded_features(self,patches,geometry,valid):
        # B x S(25) x P(5) x D -> official dimension-wise DSW embedding.
        values=patches.masked_fill(~valid[:,:,None,None],0.)
        batch=values.shape[0]
        x=self.backbone.enc_value_embedding(values.reshape(batch,125,-1))
        geo=geometry.masked_fill(~valid[:,:,None],0.)
        x=x+self.backbone.enc_pos_embedding+self.geometry(geo)[:,None,:,:]
        x=self.backbone.pre_norm(x).masked_fill(~valid[:,None,:,None],0.)
        encoded,masks=encode(self.backbone.encoder,x,valid)
        query=self.backbone.dec_pos_embedding.expand(batch,-1,-1,-1)
        return decode(self.backbone.decoder,query,encoded,masks)

    def forward(self,patches,geometry,valid):
        decoded=self.decoded_features(patches,geometry,valid)
        return self.return_head(decoded.reshape(len(decoded),-1)).squeeze(-1)

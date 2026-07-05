// paraformer_asr.h — single-header Paraformer (SAN-M encoder + CIF + SAN-M decoder) inference on ggml.
// Extracted from funasr-cpp paraformer CLI. Exposes: paraformer_load, paraformer_compute_fbank,
// paraformer_run_seg, detok_pf.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace paraformer {

static const float LN_EPS = 1e-5f;
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80,LFR_M=7,LFR_N=6;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;

static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static inline void fftc(std::vector<float>&re,std::vector<float>&im,int n){
  for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
    for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
      float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
      re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}
}

static inline std::vector<float> compute_fbank_raw(std::vector<float> wav, int& T_out){
  for(auto&v:wav)v*=32768.0f; std::vector<float> win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NB=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>> fb(NMEL,std::vector<float>(NB,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
    for(int k=0;k<NB;k++){float mf=melf(bw*k);if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1;
  std::vector<std::vector<float>> feat(T,std::vector<float>(NMEL));
  std::vector<float> re(NFFT),im(NFFT),fr(WINLEN); const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT; double mn=0;
    for(int i=0;i<WINLEN;i++)mn+=s[i]; mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;
    for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1]; fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}
    fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NB;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);
      feat[t][m]=logf(e>fl?e:fl);}}
  const int pad=(LFR_M-1)/2; int Tl=(T+LFR_N-1)/LFR_N;
  std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
  for(int i=0;i<pad;i++)pd.push_back(feat[0]);
  for(int t=0;t<T;t++)pd.push_back(feat[t]);
  while((int)pd.size()<(Tl-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
  int D=LFR_M*NMEL; std::vector<float> out((size_t)Tl*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<LFR_M;j++)
    memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
  T_out=Tl; return out;
}

// Fbank + CMVN: returns [T, 560] flat.
static inline std::vector<float> paraformer_compute_fbank(std::vector<float> wav, int& T_out,
    const float* cmvn_shift, const float* cmvn_scale){
  int F=560;
  auto fb=compute_fbank_raw(std::move(wav), T_out);
  if(T_out<1) return fb;
  for(int t=0;t<T_out;t++)
    for(int d=0;d<F;d++)
      fb[(size_t)t*F+d]=(fb[(size_t)t*F+d]+cmvn_shift[d])*cmvn_scale[d];
  return fb;
}

struct cfg {
  int d_model=512, enc_head=4, enc_blocks=50, enc_kernel=11;
  int dec_blocks=16, dec_att=16, dec3=1, dec_head=4, dec_kernel=11;
  int vocab=8404;
  float tail=0.45f, thresh=1.0f;
};

struct model {
  cfg c;
  ggml_context* ctx_w=nullptr;
  std::map<std::string, ggml_tensor*> t;
  // Cached inference resources. Populated lazily on first paraformer_run_seg call.
  // Safe to reuse across calls from a single thread at a time.
  mutable ggml_backend_t be_cache=nullptr;
  mutable ggml_gallocr_t ga_enc_cache=nullptr;
  mutable ggml_gallocr_t ga_dec_cache=nullptr;
  // Persistent encoder graph (rebuilt when T grows).
  mutable ggml_context* enc_ctx=nullptr;
  mutable ggml_tensor* enc_x=nullptr;
  mutable ggml_tensor* enc_h=nullptr;
  mutable ggml_cgraph* enc_gf=nullptr;
  mutable int enc_T_alloc=0;
  // Persistent decoder graph (rebuilt when N or T grows).
  mutable ggml_context* dec_ctx=nullptr;
  mutable ggml_tensor* dec_tgt=nullptr;
  mutable ggml_tensor* dec_mem=nullptr;
  mutable ggml_tensor* dec_out=nullptr;
  mutable ggml_cgraph* dec_gf=nullptr;
  mutable int dec_N_alloc=0;
  mutable int dec_T_alloc=0;
  ggml_tensor* g(const std::string& n){
    auto it=t.find(n);
    if(it==t.end()){fprintf(stderr,"paraformer: missing %s\n",n.c_str());return nullptr;}
    return it->second;
  }
  ~model(){
    if(enc_gf) enc_gf=nullptr;
    if(dec_gf) dec_gf=nullptr;
    if(enc_ctx) ggml_free(enc_ctx);
    if(dec_ctx) ggml_free(dec_ctx);
    if(ga_enc_cache) ggml_gallocr_free(ga_enc_cache);
    if(ga_dec_cache) ggml_gallocr_free(ga_dec_cache);
    if(be_cache) ggml_backend_free(be_cache);
  }
};

static inline ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){
  auto y=ggml_mul_mat(c,w,x); return b?ggml_add(c,y,b):y;
}
static inline ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){
  return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);
}
static inline ggml_tensor* fsmn(ggml_context*c,ggml_tensor*v,ggml_tensor*fk,int D,int T,int K){
  const int pad=(K-1)/2;
  ggml_tensor* vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0);
  ggml_tensor* acc=v;
  for(int j=0;j<K;j++){
    auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
    auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]);
    acc=ggml_add(c,acc,ggml_mul(c,ggml_cont(c,sl),wj));
  }
  return acc;
}

static inline ggml_tensor* enc_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T){
  const int D=m.c.d_model,H=m.c.enc_head,dk=D/H,K=m.c.enc_kernel;
  ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x);
  size_t nb1=qkv->nb[1];
  ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
  ggml_tensor*fm=fsmn(c,v,m.g(p+"fsmn_block.weight"),D,T,K);
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3);
  k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
  return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fm);
}

static inline ggml_tensor* enc_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool res){
  auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  auto sa=enc_attn(c,m,p+"self_attn.",h,T);
  x=res?ggml_add(c,r,sa):sa; r=x;
  h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h);
  h=ggml_relu(c,h);
  h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h);
  return ggml_add(c,r,h);
}

static inline ggml_tensor* dec_ffn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x){
  auto h=lin(c,m.g(p+"w_1.weight"),m.g(p+"w_1.bias"),x);
  h=ggml_relu(c,h);
  h=lnorm(c,h,m.g(p+"norm.weight"),m.g(p+"norm.bias"));
  return ggml_mul_mat(c,m.g(p+"w_2.weight"),h);
}

static inline ggml_tensor* cross_attn(ggml_context*c,model&m,const std::string&p,
    ggml_tensor*tgt,ggml_tensor*mem,int N,int T){
  const int D=m.c.d_model,H=m.c.dec_head,dk=D/H;
  ggml_tensor*q=lin(c,m.g(p+"linear_q.weight"),m.g(p+"linear_q.bias"),tgt);
  ggml_tensor*kv=lin(c,m.g(p+"linear_k_v.weight"),m.g(p+"linear_k_v.bias"),mem);
  size_t nb1=kv->nb[1];
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,kv,D,T,nb1,0));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,kv,D,T,nb1,(size_t)D*sizeof(float)));
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,N),0,2,1,3);
  k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,N);
  return lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o);
}

static inline ggml_tensor* dec_layer(ggml_context*c,model&m,const std::string&p,
    ggml_tensor*tgt,ggml_tensor*mem,int N,int T){
  const int D=m.c.d_model,K=m.c.dec_kernel;
  auto residual=tgt;
  auto h=lnorm(c,tgt,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  h=dec_ffn(c,m,p+"feed_forward.",h);
  auto y=lnorm(c,h,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  auto sa=fsmn(c,y,m.g(p+"self_attn.fsmn_block.weight"),D,N,K);
  auto x=ggml_add(c,residual,sa);
  residual=x;
  auto z=lnorm(c,x,m.g(p+"norm3.weight"),m.g(p+"norm3.bias"));
  auto ca=cross_attn(c,m,p+"src_attn.",z,mem,N,T);
  return ggml_add(c,residual,ca);
}

static inline ggml_tensor* dec3_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*tgt){
  auto h=lnorm(c,tgt,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  return dec_ffn(c,m,p+"feed_forward.",h);
}

static inline void add_posenc(std::vector<float>&x,int T,int depth){
  double inc=log(10000.0)/(depth/2.0-1.0);
  for(int t=0;t<T;t++){double pos=t+1;
    for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
      x[(size_t)t*depth+i]+=(float)sin(st);
      x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}

static inline bool paraformer_load(const std::string& gguf_path, model& m,
    std::vector<std::string>& vocab){
  gguf_init_params gp={false,&m.ctx_w};
  gguf_context* gg=gguf_init_from_file(gguf_path.c_str(),gp);
  if(!gg){fprintf(stderr,"paraformer: load gguf failed: %s\n",gguf_path.c_str());return false;}
  auto rdi=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  auto rdf=[&](const char*k,float d){int i=gguf_find_key(gg,k);return i<0?d:gguf_get_val_f32(gg,i);};
  m.c.enc_blocks=rdi("pf.enc.num_blocks",50);
  m.c.dec_blocks=rdi("pf.dec.num_blocks",16);
  m.c.dec_att=rdi("pf.dec.att_layer_num",16);
  m.c.dec3=rdi("pf.dec.decoders3",1);
  m.c.vocab=rdi("pf.vocab_size",8404);
  m.c.tail=rdf("pf.predictor.tail_threshold",0.45f);
  m.c.thresh=rdf("pf.predictor.threshold",1.0f);
  int ki=gguf_find_key(gg,"pf.vocab");
  if(ki>=0){int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv);
    for(int i=0;i<nv;i++){const char*s=gguf_get_arr_str(gg,ki,i); vocab[i]=s?s:"";}}
  for(int i=0;i<gguf_get_n_tensors(gg);i++){
    const char* nm=gguf_get_tensor_name(gg,i);
    m.t[nm]=ggml_get_tensor(m.ctx_w,nm);
  }
  gguf_free(gg);
  return true;
}

// Run full pipeline: fbank [T,560] -> encoder -> CIF -> decoder -> token ids.
// cmvn_shift/cmvn_scale: pointers to model's cmvn tensors. nthreads: ggml cpu threads.
static inline std::vector<int> paraformer_run_seg(const model& m,
    const std::vector<float>& fb, int T,
    const float* cmvn_shift, const float* cmvn_scale,
    int nthreads=8){
  const int D=m.c.d_model, F=560, V=m.c.vocab;
  if(T<1) return {};

  std::vector<float> feats=fb;
  for(int t=0;t<T;t++)
    for(int d=0;d<F;d++)
      feats[(size_t)t*F+d]=(feats[(size_t)t*F+d]+cmvn_shift[d])*cmvn_scale[d];

  float sc=sqrtf((float)D);
  for(auto&v:feats) v*=sc;
  add_posenc(feats,T,F);

  // Encoder
  std::vector<float> enc;
  {
    ggml_backend_t be=m.be_cache;
    ggml_gallocr_t ga=m.ga_enc_cache;
    if(!be){ be=ggml_backend_cpu_init(); m.be_cache=be; }
    if(!ga){ ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); m.ga_enc_cache=ga; }

    if(!m.enc_ctx || T > m.enc_T_alloc){
      if(m.enc_ctx){
        fprintf(stderr,"paraformer: rebuilding enc graph T %d -> %d\n", m.enc_T_alloc, T);
        ggml_free(m.enc_ctx);
      }
      ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true};
      m.enc_ctx=ggml_init(cp);
      ggml_context* c=m.enc_ctx;
      ggml_tensor* x=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,T); ggml_set_input(x);
      ggml_tensor* h=enc_layer(c,const_cast<model&>(m),"encoder.encoders0.0.",x,T,false);
      for(int i=0;i<m.c.enc_blocks-1;i++)
        h=enc_layer(c,const_cast<model&>(m),"encoder.encoders."+std::to_string(i)+".",h,T,true);
      h=lnorm(c,h,const_cast<model&>(m).g("encoder.after_norm.weight"),
                  const_cast<model&>(m).g("encoder.after_norm.bias"));
      ggml_set_output(h);
      m.enc_gf=ggml_new_graph_custom(c,32768,false);
      ggml_build_forward_expand(m.enc_gf,h);
      m.enc_x=x; m.enc_h=h; m.enc_T_alloc=T;
    }

    ggml_gallocr_alloc_graph(ga,m.enc_gf);
    ggml_backend_tensor_set(m.enc_x,feats.data(),0,ggml_nbytes(m.enc_x));
    ggml_backend_cpu_set_n_threads(be,nthreads);
    ggml_backend_graph_compute(be,m.enc_gf);
    enc.resize((size_t)T*D);
    ggml_backend_tensor_get(m.enc_h,enc.data(),0,ggml_nbytes(m.enc_h));
  }

  // CIF predictor (host): conv1d(k=3,pad=1)+residual+relu -> sigmoid -> alpha -> integrate-and-fire
  float* cw=(float*)const_cast<model&>(m).g("predictor.cif_conv1d.weight")->data;
  float* cb=(float*)const_cast<model&>(m).g("predictor.cif_conv1d.bias")->data;
  float* ow=(float*)const_cast<model&>(m).g("predictor.cif_output.weight")->data;
  float ob=((float*)const_cast<model&>(m).g("predictor.cif_output.bias")->data)[0];

  std::vector<float> outp((size_t)T*D);
  std::vector<float> alphas(T);
  for(int t=0;t<T;t++){
    for(int o=0;o<D;o++){
      float acc=cb[o];
      for(int j=0;j<3;j++){
        int tt=t+j-1; if(tt<0||tt>=T)continue;
        const float* ev=&enc[(size_t)tt*D];
        const float* wo=&cw[(size_t)o*D*3];
        for(int i=0;i<D;i++) acc+=wo[i*3+j]*ev[i];
      }
      outp[(size_t)t*D+o]=acc+enc[(size_t)t*D+o];
    }
    float a=ob;
    for(int o=0;o<D;o++){float r=outp[(size_t)t*D+o]; if(r<0)r=0; a+=ow[o]*r;}
    float s=1.0f/(1.0f+expf(-a)); alphas[t]=s>0?s:0;
  }

  std::vector<float> hid=enc;
  hid.resize((size_t)(T+1)*D,0.0f);
  std::vector<float> al=alphas;
  al.push_back(m.c.tail);
  int L=T+1;

  std::vector<float> acoustic;
  acoustic.reserve(64*D);
  float integrate=0;
  std::vector<float> frame(D,0.0f);
  for(int t=0;t<L;t++){
    float alpha=al[t];
    float dc=1.0f-integrate;
    integrate+=alpha;
    bool fire=integrate>=m.c.thresh;
    float cur=fire?dc:alpha;
    float rem=alpha-cur;
    for(int d=0;d<D;d++) frame[d]+=cur*hid[(size_t)t*D+d];
    if(fire){
      acoustic.insert(acoustic.end(),frame.begin(),frame.end());
      integrate-=1.0f;
      for(int d=0;d<D;d++) frame[d]=rem*hid[(size_t)t*D+d];
    }
  }
  int N=(int)acoustic.size()/D;
  if(N<1) return {};

  // Decoder
  std::vector<float> logits;
  {
    ggml_backend_t be=m.be_cache;
    ggml_gallocr_t ga=m.ga_dec_cache;
    if(!be){ be=ggml_backend_cpu_init(); m.be_cache=be; }
    if(!ga){ ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); m.ga_dec_cache=ga; }

    if(!m.dec_ctx || N > m.dec_N_alloc || T > m.dec_T_alloc){
      if(m.dec_ctx){
        fprintf(stderr,"paraformer: rebuilding dec graph (N,T) (%d,%d) -> (%d,%d)\n",
                m.dec_N_alloc, m.dec_T_alloc, N, T);
        ggml_free(m.dec_ctx);
      }
      ggml_init_params cp={(size_t)2048*1024*1024,nullptr,true};
      m.dec_ctx=ggml_init(cp);
      ggml_context* c=m.dec_ctx;
      ggml_tensor* tgt=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,N); ggml_set_input(tgt);
      ggml_tensor* mem=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,T); ggml_set_input(mem);
      ggml_tensor* x=tgt;
      for(int i=0;i<m.c.dec_att;i++)
        x=dec_layer(c,const_cast<model&>(m),"decoder.decoders."+std::to_string(i)+".",x,mem,N,T);
      for(int i=0;i<m.c.dec3;i++)
        x=dec3_layer(c,const_cast<model&>(m),"decoder.decoders3."+std::to_string(i)+".",x);
      x=lnorm(c,x,const_cast<model&>(m).g("decoder.after_norm.weight"),
                  const_cast<model&>(m).g("decoder.after_norm.bias"));
      x=lin(c,const_cast<model&>(m).g("decoder.output_layer.weight"),
              const_cast<model&>(m).g("decoder.output_layer.bias"),x);
      ggml_set_output(x);
      m.dec_gf=ggml_new_graph_custom(c,32768,false);
      ggml_build_forward_expand(m.dec_gf,x);
      m.dec_tgt=tgt; m.dec_mem=mem; m.dec_out=x;
      m.dec_N_alloc=N; m.dec_T_alloc=T;
    }

    ggml_gallocr_alloc_graph(ga,m.dec_gf);
    ggml_backend_tensor_set(m.dec_tgt,acoustic.data(),0,ggml_nbytes(m.dec_tgt));
    ggml_backend_tensor_set(m.dec_mem,enc.data(),0,ggml_nbytes(m.dec_mem));
    ggml_backend_cpu_set_n_threads(be,nthreads);
    ggml_backend_graph_compute(be,m.dec_gf);
    logits.resize((size_t)V*N);
    ggml_backend_tensor_get(m.dec_out,logits.data(),0,ggml_nbytes(m.dec_out));
  }

  // Argmax
  std::vector<int> seg_ids;
  seg_ids.reserve(N);
  for(int n=0;n<N;n++){
    const float* col=&logits[(size_t)n*V];
    int am=0; float best=col[0];
    for(int v=1;v<V;v++) if(col[v]>best){best=col[v];am=v;}
    seg_ids.push_back(am);
  }
  return seg_ids;
}

static inline std::string pf_trim(const std::string&s){
  size_t a=s.find_first_not_of(' ');
  if(a==std::string::npos) return "";
  size_t b=s.find_last_not_of(' ');
  return s.substr(a,b-a+1);
}

static inline std::string detok_pf(const std::vector<int>& ids,
    const std::vector<std::string>& vocab){
  std::string s;
  for(int id:ids){
    if(id==1||id==2) continue;
    if(id>=0&&(int)id<(int)vocab.size()) s+=vocab[id];
  }
  size_t p;
  while((p=s.find("@@"))!=std::string::npos) s.erase(p,2);
  const std::string lb="\xe2\x96\x81";
  while((p=s.find(lb))!=std::string::npos) s.replace(p,3," ");
  return pf_trim(s);
}

} // namespace paraformer

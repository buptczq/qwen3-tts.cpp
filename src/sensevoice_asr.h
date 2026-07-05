// sensevoice.h — single-header SenseVoiceSmall (SAN-M encoder + CTC) inference on ggml.
// Exposes: sensevoice_load, compute_fbank (+ optional cached mel filterbank), run_seg, detok_sv.
// No IO / no stdout — pure model API for CLI and shared-lib consumers.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#ifdef __APPLE__
#include <sys/resource.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sensevoice {

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

// Precomputed 80-mel filterbank [NMEL][NFFT/2+1]. Build once, pass to compute_fbank.
static inline std::vector<std::vector<float>> mel_filterbank(){
  const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>> fb(NMEL,std::vector<float>(NBIN,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
    for(int k=0;k<NBIN;k++){float mf=melf(bw*k); if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  return fb;
}

// 80-log-mel + LFR(m=7,n=6) -> [Tl, 560] flat. mel_fb optional cache (nullptr = rebuild).
static inline std::vector<float> compute_fbank(std::vector<float> wav,int&T_out,
                                               const std::vector<std::vector<float>>* mel_fb=nullptr){
  for(auto&v:wav)v*=32768.0f; std::vector<float> win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  std::vector<std::vector<float>> fb0; const std::vector<std::vector<float>>* fb=mel_fb;
  if(!fb){ fb0=mel_filterbank(); fb=&fb0; }
  const int NBIN=NFFT/2+1;
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; if(T<1)T=0;
  std::vector<std::vector<float>> feat(T,std::vector<float>(NMEL));
  std::vector<float> re(NFFT),im(NFFT),fr(WINLEN); const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT; double mn=0; for(int i=0;i<WINLEN;i++)mn+=s[i]; mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn; for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1]; fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;} fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if((*fb)[m][k]>0)e+=(*fb)[m][k]*(re[k]*re[k]+im[k]*im[k]); feat[t][m]=logf(e>fl?e:fl);}}
  const int pad=(LFR_M-1)/2; int Tl=(T+LFR_N-1)/LFR_N;
  if(T<1){T_out=0; return {};} // too short for one frame
  std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
  for(int i=0;i<pad;i++)pd.push_back(feat[0]); for(int t=0;t<T;t++)pd.push_back(feat[t]);
  while((int)pd.size()<(Tl-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
  int D=LFR_M*NMEL; std::vector<float> out((size_t)Tl*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<LFR_M;j++)memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
  T_out=Tl; return out;
}

struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,vocab=25055,blank=0; };
struct model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
  // Cached inference resources. Populated lazily on first run_seg call.
  // Safe to reuse across calls from a single thread at a time.
  mutable ggml_backend_t be_cache=nullptr;
  mutable ggml_gallocr_t ga_cache=nullptr;
  // Persistent threadpool — avoids creating/destroying 8 threads on every
  // ggml_backend_graph_compute call, which is extremely expensive on macOS.
  mutable ggml_threadpool_t threadpool=nullptr;
  mutable int threadpool_nthreads=0;
  // Persistent compute graph (built once, reused across calls).
  // Rebuilt only when the time dimension N grows beyond N_alloc.
  mutable ggml_context* graph_ctx=nullptr;
  mutable ggml_tensor* x_p=nullptr;
  mutable ggml_tensor* logits_p=nullptr;
  mutable ggml_cgraph* gf_p=nullptr;
  mutable int N_alloc=0;
  mutable int run_counter=0;  // incremented on each run_seg call
  ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());return nullptr;}return it->second;}
  ~model(){
    if(gf_p) { gf_p=nullptr; }  // owned by graph_ctx
    if(graph_ctx) ggml_free(graph_ctx);
    if(threadpool) ggml_threadpool_free(threadpool);
    if(ga_cache) ggml_gallocr_free(ga_cache);
    if(be_cache) ggml_backend_free(be_cache);
  }
};

static inline ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static inline ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static inline ggml_tensor* sanm_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T){
  const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
  ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
  ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
  const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
  ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
  for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
    auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
  return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static inline ggml_tensor* sanm_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool res){
  auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
  h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
  h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static inline void add_posenc(std::vector<float>&x,int T,int depth){
  double inc=log(10000.0)/(depth/2.0-1.0);
  for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
    x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}

// Load GGUF: weights + cfg + query tokens + vocab. Returns false on failure (no exit).
static inline bool sensevoice_load(const std::string& gguf_path, model& m,
                                   std::vector<int>& qtok, std::vector<std::string>& vocab){
  gguf_init_params gp={false,&m.ctx_w}; gguf_context*gg=gguf_init_from_file(gguf_path.c_str(),gp);
  if(!gg){fprintf(stderr,"sensevoice: load gguf failed: %s\n",gguf_path.c_str());return false;}
  auto rd=[&](const char*k,int d)->int{int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  m.c.d_model=rd("sv.output_size",512); m.c.n_head=rd("sv.attention_heads",4);
  m.c.num_blocks=rd("sv.num_blocks",50); m.c.tp_blocks=rd("sv.tp_blocks",20);
  m.c.kernel=rd("sv.kernel_size",11); m.c.vocab=rd("sv.vocab_size",25055); m.c.blank=rd("sv.blank_id",0);
  int qi=gguf_find_key(gg,"sv.query_tokens"); int nq=qi<0?0:(int)gguf_get_arr_n(gg,qi);
  qtok.resize(nq); for(int i=0;i<nq;i++) qtok[i]=((const int32_t*)gguf_get_arr_data(gg,qi))[i];
  int ki=gguf_find_key(gg,"sv.vocab"); if(ki>=0){int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv);
    for(int i=0;i<nv;i++){const char*s=gguf_get_arr_str(gg,ki,i); vocab[i]=s?s:"";}}
  for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
  gguf_free(gg);
  return true;
}

// Run encoder+CTC on fbank [T,560]; returns greedy-CTC token ids (collapsed, blank dropped).
// emb = embed.weight data ptr ([16,560] row-major); nthreads = ggml cpu threads.
static inline std::vector<int> run_seg(const model& m, const int* qtok, int nq,
                                       const float* emb, const std::vector<float>& fb, int T,
                                       int nthreads=8){
  const int F=560, D=m.c.d_model, V=m.c.vocab;
  if(T<1) return {};
  int N=nq+T;
  m.run_counter++;
  int call_id=m.run_counter;

  // RSS memory (macOS)
  size_t rss_kb=0;
#ifdef __APPLE__
  { struct rusage ru; getrusage(RUSAGE_SELF,&ru); rss_kb=(size_t)ru.ru_maxrss/1024; }
#endif

  ggml_backend_t be=m.be_cache;
  ggml_gallocr_t ga=m.ga_cache;
  if(!be){ be=ggml_backend_cpu_init(); m.be_cache=be; fprintf(stderr,"sensevoice: [call#%d] CPU backend created\n",call_id); }
  if(!ga){ ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); m.ga_cache=ga; }
  // Create or resize persistent threadpool to avoid per-call thread create/destroy
  if(!m.threadpool || m.threadpool_nthreads != nthreads){
    if(m.threadpool) ggml_threadpool_free(m.threadpool);
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nthreads);
    m.threadpool = ggml_threadpool_new(&tpp);
    m.threadpool_nthreads = nthreads;
    ggml_backend_cpu_set_threadpool(be, m.threadpool);
    fprintf(stderr,"sensevoice: [call#%d] threadpool created nthreads=%d\n",call_id,nthreads);
  }

  auto t_us=[](){auto tp=std::chrono::steady_clock::now();return std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();};
  int64_t t0=t_us();

  // Build (or rebuild if N grew) the persistent compute graph.
  bool rebuilt=false;
  bool is_rebuild = (m.graph_ctx != nullptr);
  if(!m.graph_ctx || N > m.N_alloc){
    if(m.graph_ctx){
      fprintf(stderr,"sensevoice: rebuilding graph ctx N %d -> %d\n", m.N_alloc, N);
      ggml_free(m.graph_ctx);
    }
    ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true};
    m.graph_ctx=ggml_init(cp);
    ggml_context*c=m.graph_ctx;
    ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,N); ggml_set_input(x);
    ggml_tensor*h=sanm_layer(c,const_cast<model&>(m),"encoder.encoders0.0.",x,N,false);
    for(int i=0;i<m.c.num_blocks-1;i++) h=sanm_layer(c,const_cast<model&>(m),"encoder.encoders."+std::to_string(i)+".",h,N,true);
    h=lnorm(c,h,const_cast<model&>(m).g("encoder.after_norm.weight"),const_cast<model&>(m).g("encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) h=sanm_layer(c,const_cast<model&>(m),"encoder.tp_encoders."+std::to_string(i)+".",h,N,true);
    h=lnorm(c,h,const_cast<model&>(m).g("encoder.tp_norm.weight"),const_cast<model&>(m).g("encoder.tp_norm.bias"));
    ggml_tensor*logits=lin(c,const_cast<model&>(m).g("ctc.ctc_lo.weight"),const_cast<model&>(m).g("ctc.ctc_lo.bias"),h);
    ggml_set_output(logits);
    m.gf_p=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(m.gf_p,logits);
    m.x_p=x; m.logits_p=logits; m.N_alloc=N;
    // Re-create gallocr for rebuilt graph to avoid stale memory allocation
    if(is_rebuild && m.ga_cache){ ggml_gallocr_free(m.ga_cache); m.ga_cache=nullptr; }
    if(!m.ga_cache){ ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); m.ga_cache=ga; }
    rebuilt=true;
  }
  int64_t t1=t_us();

  int Na=m.N_alloc; // tensor may be larger than N if graph was reused
  std::vector<float> inp((size_t)Na*F, 0.0f); // zero-padded to N_alloc
  for(int i=0;i<nq;i++) memcpy(&inp[(size_t)i*F], &emb[qtok[i]*F], F*sizeof(float));
  memcpy(&inp[(size_t)nq*F], fb.data(), (size_t)T*F*sizeof(float));
  // Pad trailing rows with a copy of the last real frame (instead of zeros)
  // to avoid boundary artifacts from the SAN-M FSMN conv-left kernel.
  if(Na > N){
    const float* last_row = &inp[(size_t)(N-1)*F];
    for(int i=N;i<Na;i++) memcpy(&inp[(size_t)i*F], last_row, F*sizeof(float));
  }
  float sc=sqrtf((float)D); for(int i=0;i<N*F;i++) inp[i]*=sc;
  add_posenc(inp,N,F); // posenc only for real N positions

  ggml_gallocr_alloc_graph(ga,m.gf_p);
  int64_t t2=t_us();
  ggml_backend_tensor_set(m.x_p,inp.data(),0,ggml_nbytes(m.x_p));
  // Threadpool is persistent (set via ggml_backend_cpu_set_threadpool at creation).
  // No need to call set_n_threads every call.
  int64_t t3=t_us();
  std::vector<int> seg_ids;
  if(ggml_backend_graph_compute(be,m.gf_p)==GGML_STATUS_SUCCESS){
    int64_t t4=t_us();
    std::vector<float> lg((size_t)V*Na); ggml_backend_tensor_get(m.logits_p,lg.data(),0,ggml_nbytes(m.logits_p));
    int prev=-1;
    for(int n=0;n<N;n++){ const float*col=&lg[(size_t)n*V]; int am=0; float best=col[0];
      for(int v=1;v<V;v++) if(col[v]>best){best=col[v];am=v;}
      if(am!=prev && am!=m.c.blank) seg_ids.push_back(am); prev=am; }
    size_t rss2_kb=0;
#ifdef __APPLE__
    { struct rusage ru; getrusage(RUSAGE_SELF,&ru); rss2_kb=(size_t)ru.ru_maxrss/1024; }
#endif
    fprintf(stderr,"sensevoice: [call#%d] N=%d Na=%d%s nthreads=%d graph_build=%.1fms alloc=%.1fms set_inp=%.1fms compute=%.1fms get_out=%.1fms total=%.1fms rss_before=%zuMB rss_after=%zuMB\n",
      call_id, N, Na, rebuilt?" REBUILT":"", nthreads, (t1-t0)/1000.0, (t2-t1)/1000.0, (t3-t2)/1000.0, (t4-t3)/1000.0, (t_us()-t4)/1000.0, (t_us()-t0)/1000.0, rss_kb, rss2_kb);
  } else { fprintf(stderr,"sensevoice: [call#%d] compute failed\n",call_id); }
  return seg_ids;
}

static inline std::string sv_trim(const std::string&s){size_t a=s.find_first_not_of(' ');if(a==std::string::npos)return "";size_t b=s.find_last_not_of(' ');return s.substr(a,b-a+1);}
// SentencePiece piece join; U+2581 ("▁") -> space; <|...|> meta skipped unless keep_tags.
static inline std::string detok_sv(const std::vector<int>&ids,const std::vector<std::string>&vocab,bool keep_tags){
  std::string s; for(int id:ids){ if(id<0||id>=(int)vocab.size())continue; const std::string&p=vocab[id];
    if(!keep_tags && p.size()>=2 && p[0]=='<' && p[1]=='|') continue;
    s+=p; }
  const std::string lb="\xe2\x96\x81"; size_t pp; while((pp=s.find(lb))!=std::string::npos)s.replace(pp,3," ");
  return sv_trim(s);
}

} // namespace sensevoice

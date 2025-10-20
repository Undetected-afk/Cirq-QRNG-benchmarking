import time, math, numpy as np, cirq
from scipy.stats import chisquare

def build_qrng_circuit(): return cirq.Circuit(cirq.H(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0), key='m'))

def sample_qrng_bits_cirq(n_bits, simulator=None):
    if simulator is None: simulator = cirq.Simulator()
    c = build_qrng_circuit()
    r = simulator.run(c, repetitions=n_bits)
    return r.measurements['m'].reshape(-1).astype(np.uint8)

def numpy_bits(n_bits): return np.random.randint(0, 2, n_bits, dtype=np.uint8)

def monobit_frequency(b): return float(b.mean())
def min_entropy(b): p = b.mean(); q = 1-p; return -math.log2(max(p,q)) if p not in [0,1] else 0
def lag1_autocorr(b): x = 2*b.astype(float)-1; return float(np.corrcoef(x[:-1],x[1:])[0,1])
def runs(b): return sum(b[i]!=b[i-1] for i in range(1,len(b)))+1
def chi_bytes(b):
    n=len(b); pad=(-n)%8
    if pad: b=np.concatenate([b,np.zeros(pad,dtype=np.uint8)])
    by=np.packbits(b); c=np.bincount(by,minlength=256); e=np.ones(256)*(c.sum()/256)
    s,p=chisquare(c,e); return s,p

def bench(f,n,label):
    for i in n:
        t=time.perf_counter();b=f(i);d=time.perf_counter()-t
        print(f"{label} {i}: {d:.4f}s {(i/d):,.0f} bits/s")
    return b

if __name__=="__main__":
    s=cirq.Simulator()
    for f,l in [(lambda n:sample_qrng_bits_cirq(n,s),'Cirq'),(numpy_bits,'NumPy')]:
        bench(f,[1000,10000,100000],l)
    b=sample_qrng_bits_cirq(100000,s)
    p=monobit_frequency(b);h=min_entropy(b);a=lag1_autocorr(b);r=runs(b);c,pv=chi_bytes(b)
    print(f"p(1)={p:.4f} Hmin={h:.4f} autocorr={a:.4f} runs={r} chi2={c:.2f} p={pv:.4f}")

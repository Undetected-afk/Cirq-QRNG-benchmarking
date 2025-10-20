import time, math, numpy as np, cirq, matplotlib.pyplot as plt
from scipy.stats import chisquare

def build_qrng_circuit():
    q = cirq.LineQubit(0)
    return cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))

def sample_qrng_bits(n_bits, simulator=None):
    if simulator is None: simulator = cirq.Simulator()
    c = build_qrng_circuit()
    result = simulator.run(c, repetitions=n_bits)
    return result.measurements['m'].reshape(-1).astype(np.uint8)

def numpy_bits(n_bits): return np.random.randint(0, 2, n_bits, dtype=np.uint8)

def monobit_frequency(b): return float(b.mean())
def min_entropy(b): p=b.mean(); q=1-p; return -math.log2(max(p,q)) if 0<p<1 else 0
def lag1_autocorr(b):
    if len(b)<2: return 0.0
    x=2*b.astype(float)-1
    return float(np.corrcoef(x[:-1],x[1:])[0,1])
def runs(b): return sum(b[i]!=b[i-1] for i in range(1,len(b)))+1
def chi_bytes(b):
    n=len(b); pad=(-n)%8
    if pad: b=np.concatenate([b,np.zeros(pad,dtype=np.uint8)])
    by=np.packbits(b); c=np.bincount(by,minlength=256); e=np.ones(256)*(c.sum()/256)
    s,p=chisquare(c,e); return s,p

def benchmark(f,label):
    sizes=[1_000,10_000,100_000]
    tms=[]
    for n in sizes:
        t=time.perf_counter()
        b=f(n)
        d=time.perf_counter()-t
        tms.append((n,d))
        print(f"{label} {n}: {d:.4f}s {(n/d):,.0f} bits/s")
    return tms

def plot_bench(cirq_times,numpy_times):
    plt.figure()
    plt.plot([n for n,_ in cirq_times],[n/t for n,t in cirq_times],'o-',label='Cirq QRNG')
    plt.plot([n for n,_ in numpy_times],[n/t for n,t in numpy_times],'o-',label='NumPy PRNG')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Bits Generated (log)'); plt.ylabel('Throughput (bits/s, log)')
    plt.title('QRNG vs PRNG Benchmark')
    plt.legend(); plt.tight_layout(); plt.show()

def summary(bits):
    p=monobit_frequency(bits); h=min_entropy(bits); a=lag1_autocorr(bits)
    r=runs(bits); c,pv=chi_bytes(bits)
    print(f"p(1)={p:.4f}\nHmin={h:.4f}\nautocorr={a:.4f}\nruns={r}\nchi2={c:.2f}, p={pv:.4f}")

if __name__=="__main__":
    sim=cirq.Simulator()
    print("=== Benchmarking Cirq QRNG ===")
    cirq_t=benchmark(lambda n:sample_qrng_bits(n,sim),"Cirq")
    print("\n=== Benchmarking NumPy PRNG ===")
    np_t=benchmark(numpy_bits,"NumPy")
    plot_bench(cirq_t,np_t)
    print("\n=== Quality (Cirq 100k bits) ===")
    bits=sample_qrng_bits(100_000,sim)
    summary(bits)

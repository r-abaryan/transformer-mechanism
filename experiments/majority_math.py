"""Exact integer witnesses for majority, sensitivity and uniform attention."""
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    records=[]; strings_checked=0; neighbor_checks=0
    for n in (1,3,5,7,9,11):
        flips_total=0; boundary_counts=set(); last_errors=0; uniform_errors=0
        for bits in itertools.product((0,1),repeat=n):
            signed=sum(2*b-1 for b in bits); target=int(signed>0)
            margin=abs(signed); count=0
            for j in range(n):
                neighbor=bits[:j]+(1-bits[j],)+bits[j+1:]
                count+=int(int(sum(neighbor)>n//2)!=target); neighbor_checks+=1
            assert count==((n+1)//2 if margin==1 else 0)
            if margin==1: boundary_counts.add(count)
            # n signed values and a zero-valued marker under uniform attention.
            uniform_output=Fraction(signed,n+1)
            uniform_errors+=int(int(uniform_output>0)!=target)
            last_errors+=int(bits[-1]!=target); flips_total+=count; strings_checked+=1
        actual=Fraction(flips_total,2**n)
        analytic=Fraction(n*math.comb(n-1,(n-1)//2),2**(n-1))
        assert actual==analytic and uniform_errors==0
        last_accuracy=Fraction(2**n-last_errors,2**n)
        assert last_accuracy==Fraction(1,2)+Fraction(math.comb(n-1,(n-1)//2),2**n)
        records.append(dict(n=n,strings=2**n,mean_flips=str(actual),boundary_flips=sorted(boundary_counts),
                            mean_flips_float=float(actual),single_bit_accuracy=str(last_accuracy),uniform_attention_errors=uniform_errors))
    out=ROOT/'results/majority_math.json'; assert not out.exists()
    result=dict(passed=True,strings_checked=strings_checked,neighbor_checks=neighbor_checks,cases=records,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Integer/rational majority identities and an abstract uniform-attention head, not a trained network mechanism')
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()

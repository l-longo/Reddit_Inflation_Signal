"""
new_codes_real_time_llama70B.py  --  real-time nowcasts from the Llama-3.3-70B signals
======================================================================================

Terminal version of nowcast_codes/new_codes_real_time_llama70B.ipynb.

The estimation loop is the one in new_codes_real_time.py -- the two notebooks
differ only in which daily signal file they read and in how the output files are
named -- so it is imported rather than duplicated.

Usage
-----
    python new_codes_real_time_llama70B.py                     # core PCE
    python new_codes_real_time_llama70B.py --target CPIAUCSL
    python new_codes_real_time_llama70B.py --cutoffs 22 --save
"""

from new_codes_real_time import build_parser, main

if __name__ == "__main__":
    main(build_parser(signals="llama70b").parse_args())

# BrainIAC preprocessing v2 test fixtures

All fixtures for `tests/test_brainiac_preprocessing_v2.py` are **deidentified
synthetic phantoms** generated deterministically at test time by the
`_synthetic_t1` helper inside the test module (tiny float32 ellipsoids in a
12×14×16 or 8×8×8 lattice, identity affine). No study data, no subject
identifiers, no labels, and no PET material are read or stored here.

This directory intentionally contains no binary artifacts; keeping fixtures
generative guarantees they can never leak real data and makes every test run
reproducible from source alone.

def test_import_version():
    import nvidia_arch
    assert hasattr(nvidia_arch, '__version__')

def test_print_summary():
    from nvidia_arch import print_summary
    print_summary(min_sm=60)

def test_get_architectures_basic():
    from nvidia_arch import get_architectures
    result = get_architectures(cuda_ver='13.0', min_sm=75)
    assert isinstance(result, list) or isinstance(result, tuple)

def test_get_architectures_cons_jets():
    from nvidia_arch import get_architectures
    result = get_architectures(gpu_type='cons', cuda_ver='13.0', min_sm=75, return_mode='cc_string', add_ptx=True)
    assert isinstance(result, str)
    result = get_architectures(gpu_type='cons+jets', cuda_ver='13.2', min_sm=75, return_mode='sm_list', add_ptx=True)
    assert isinstance(result, list)
    result = get_architectures(gpu_type='jets', cuda_ver='12.8', min_sm=60, return_mode='cc_list', add_ptx=True)
    assert isinstance(result, list)

def test_get_architectures_make_gencode_flags():
    from nvidia_arch import get_architectures, make_gencode_flags
    arches = get_architectures(gpu_type='jets', cuda_ver='13.0', min_sm=75)
    flags = make_gencode_flags(arches, add_ptx=True)
    assert isinstance(flags, list)

def test_detect_ctk():
    from nvidia_arch import detect_ctk
    res = detect_ctk()
    # expecting None on non-CTK environments like basic CI, or dict if ctk is installed
    assert res is None or isinstance(res, dict)

def test_find_gpu():
    from nvidia_arch import find_gpu
    # Should not throw even if no GPU is present
    res = find_gpu(extra_query_gpu='serial,temperature.gpu')
    # Expecting None if no GPU is found, or a list of dicts if GPUs are found
    assert res is None or isinstance(res, list)

if __name__ == "__main__":
    test_import_version()
    test_print_summary()
    test_get_architectures_basic()
    test_get_architectures_cons_jets()
    test_get_architectures_make_gencode_flags()
    test_detect_ctk()
    test_find_gpu()
    print("All basic workflow tests passed!")

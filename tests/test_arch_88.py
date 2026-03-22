import nvidia_arch as na

CUDA_VERSIONS_STR = sorted(na.CUDA_FILTERS.keys())
CUDA_VERSIONS_FLOAT = [float(v) for v in CUDA_VERSIONS_STR]

def test_arch_88():
    """
    Test get_architectures function to see if it correctly lists the correct architectures.
    Architecture 88 or 8.8 is mysterious; it is not documented or explained anywhere prior 
    to CUDA 13.0, but it appears as supported starting from CUDA 13.0.
    """
    for cuda_ver in CUDA_VERSIONS_FLOAT:

        # cc_string mode
        arches = na.get_architectures(cuda_ver=cuda_ver, return_mode='cc_string')
        if cuda_ver < 13.0 and '8.8' in arches:
            # This is unexpected, as 8.8 should not be supported in CUDA versions prior to 13.0
            raise AssertionError(f"CUDA {cuda_ver} should not support architecture 8.8, but it is listed in {arches}")
        if cuda_ver >= 13.0 and '8.8' not in arches:
            # This is unexpected, as 8.8 should be supported in CUDA versions 13.0 and later
            raise AssertionError(f"CUDA {cuda_ver} should support architecture 8.8, but it is not listed in {arches}")
        
        # sm_list mode
        arches = na.get_architectures(cuda_ver=cuda_ver, return_mode='sm_list')
        if cuda_ver < 13.0 and '88' in arches:
            # This is unexpected, as 8.8 should not be supported in CUDA versions prior to 13.0
            raise AssertionError(f"CUDA {cuda_ver} should not support architecture 8.8, but it is listed in {arches}")
        if cuda_ver >= 13.0 and '88' not in arches:
            # This is unexpected, as 8.8 should be supported in CUDA versions 13.0 and later
            raise AssertionError(f"CUDA {cuda_ver} should support architecture 8.8, but it is not listed in {arches}")

    print("test_arch_88 passed successfully for all CUDA versions.")


if __name__ == "__main__":
    test_arch_88()

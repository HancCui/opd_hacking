from setuptools import find_packages, setup


def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


# Setup configuration
setup(
    author="slime Team",
    name="opd-hacking",
    version="0.2.2",
    packages=find_packages(include=["slime", "slime.*"]),
    package_data={
        "slime.backends.megatron_utils": ["kernels/*.py", "kernels/int4_qat/*.py", "kernels/int4_qat/*.cu"]
    },
    include_package_data=True,
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)

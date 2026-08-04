"""
Environment Check

Purpose:
Verify that PyTorch can detect the GPU and report
basic hardware information before we start loading LLMs.
"""

import platform
import torch


def main() -> None:
    print("=" * 60)
    print("ETL LLM Fine-Tuning Environment Check")
    print("=" * 60)

    print(f"Python Version : {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version   : {torch.version.cuda}")

        props = torch.cuda.get_device_properties(0)

        print(f"Total VRAM     : {props.total_memory / 1024**3:.2f} GB")
        print(f"Multi Processor: {props.multi_processor_count}")
        print(f"Device Index   : {torch.cuda.current_device()}")

    else:
        print("❌ CUDA is NOT available.")

    print("=" * 60)


if __name__ == "__main__":
    main()
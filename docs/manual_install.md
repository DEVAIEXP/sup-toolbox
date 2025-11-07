# Installation Guide

This guide provides different installation methods for **SUP-Toolbox**. Please choose the one that best fits your needs.

## Prerequisites

Before you begin, ensure you have the following on your system:
-   **Python 3.10 or newer.**
-   **Git** (required for the Developer installation).
-   **NVIDIA GPU with CUDA drivers.** For optimal performance, a compatible CUDA toolkit is required.

---

## For Users (Recommended)

This method is for users who want to use the `sup-toolbox` command-line tool without modifying its source code.

### Step 1: Create and Activate a Virtual Environment

It is strongly recommended to use a virtual environment to avoid conflicts with other Python projects.

1.  **Create the environment:**
    ```bash
    python3 -m venv venv
    ```
    *(On Windows, you might use `python` instead of `python3`)*

2.  **Activate the environment:**
    *   **Windows (Command Prompt):** `.\venv\Scripts\activate.bat`
    *   **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
    *   **Linux / macOS:** `source venv/bin/activate`

### Step 2: Install PyTorch with CUDA Support

The version of PyTorch listed in our dependencies is the generic one and may not include CUDA support. To ensure GPU acceleration, you must install the correct PyTorch version for your system **before** installing `sup-toolbox`.

1.  Visit the official **[PyTorch website](https://pytorch.org/get-started/locally/)**.
2.  Select your OS, Package Manager (`pip`), Language (`Python`), and the appropriate CUDA version.
3.  Copy and run the generated command.

    **Example for Windows/Linux with CUDA 12.6:**
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

### Step 3: Install SUP-Toolbox

Now, you can install the `sup-toolbox` package directly from its repository using `pip`. This will automatically handle all other dependencies.

```bash
pip install git+https://github.com/DEVAIEXP/sup-toolbox.git
```

### Verification

After installation, you can verify that the command-line tool is available:
```bash
sup-toolbox --help
```
This should display the main help message for the CLI. You are now ready to use the tool.

---

## For Developers (from Source)

This method is for developers who want to modify or contribute to the `sup-toolbox` source code.

### Step 1: Clone the Repository

```bash
git clone https://github.com/DEVAIEXP/sup-toolbox.git
cd sup-toolbox
```

### Step 2: Run the Automated Setup Script

The easiest way to set up a development environment is to use the provided scripts. They will automatically create a virtual environment, activate it, and install all dependencies.

*   **On Windows:**
    ```batch
    .\setup.bat
    ```

*   **On Linux or macOS:**
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

### Manual Setup (Alternative for Developers)

If you prefer to set up the environment manually:

1.  **Create and Activate a Virtual Environment** (see instructions in the "For Users" section above).

2.  **Install PyTorch with CUDA Support** (see instructions in the "For Users" section above). This step is crucial.

3.  **Install the Package in Editable Mode:**
    This command links the installed package to your source code, so any changes you make are immediately reflected.
    ```bash
    pip install -e .
    ```

### Verification

After setting up, verify that the CLI command works and points to your local code:
```bash
sup-toolbox --help
```
You can now edit the source code in the `sup-toolbox` directory, and your changes will be live the next time you run the command.

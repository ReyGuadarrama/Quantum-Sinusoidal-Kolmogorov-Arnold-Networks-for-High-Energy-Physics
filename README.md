# Quantum Sinusoidal-Kolmogorov-Arnold-Networks-for-High-Energy-Physics
The purpose of the repository is write the code develop for the project QKAN for GSoC at ML4SCI in 2026

---

## Enviroment
Follow the next comand to install. 

```bash
pip install uv
```

Create the virtual enviroment
```bash
uv sync
```

---

## Structure

The current file includes
* notebooks: dir with initial exploration
* src: dir with pipelines

---

## Download datasets
To improv the time of download we use aria2. We also need unzip to Higgs file.

Before running the download script, ensure you have `aria2` and `unzip` installed on your WSL/Linux environment. 

### Installation on Ubuntu/Debian (WSL):
Run the following command in your terminal:

```bash
sudo apt update && sudo apt install -y aria2 unzip
```

Once the prerequisites are installed, make the script executable and run it:
```bash
chmod +x download_data.sh
./download_data.sh
```

### Resuming interrupted downloads
If the download script fails or is interrupted due to network issues, you can safely run it again:
```bash
./download_data.sh
```

* Resuming: aria2 automatically handles partial downloads and will resume from where it left off.

* Zenodo Edge Case: If the script successfully downloaded and renamed files like train.h5 or test.h5 before interrupting, running the script again might re-download them because the clean filenames no longer match the source URL query. If you want to avoid this, you can comment out the completed URLs inside the script before running it again, or simply let aria2 overwrite them. 

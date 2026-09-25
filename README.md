# ml_env
<!-- https://github.com/ldan-dev/ml_env -->

<br />
<div align="center">
  <a href="https://github.com/ldan-dev/ml_env">
    <img src="https://github.com/ldan-dev/assets/blob/main/logo_ug.png?raw=true" alt="Logo UG" width="250" >
  </a>

  <h3 align="center">ml_env</h3>
  <p align="center">
     Licenciatura en Ingeniería de Datos e Inteligencia Artificial (IS75LI0801)
    <br />
    Universidad de Guanajuato - Campus Irapuato-Salamanca
    <br />
    <br />
    <img src="https://img.shields.io/badge/python-3.13-blue.svg" alt="Python Version">
  </p>
</div>




## Index
- [Concepts](#concepts)
- [Resources](#resources)
- [Clone the repository](#clone-the-repository)
- [Environment Setup](#environment-setup)
- [Contributing](#contributing)
- [Contact](#contact)
- [Use of the repository](#use-of-the-repository)

# Concepts
- concept

# Resources
- [text](link)
- [text](link)



# Contact
- [Email 1](mailto:ld.avinaneri@ugto.mx)
- [Email 2](mailto:daniel.avina.neri@gmail.com)
- [Github](https://github.com/ldan-dev)

# Clone the repository
To clone this repository, use the following command:
```bash
git clone https://github.com/ldan-dev/ml_env.git
```

# Environment Setup

## a) (Mini)Conda environment:

- Creation:
```bash
conda create -n ml_env python=3.13 -y
```
- Activation:
```bash
conda activate ml_env
```

- Installation of dependencies:
```bash
conda install --file requirements.txt -y
```

## b) Without conda:

- Create a virtual environment:
```bash
python -m venv ml_env 
```
- Activation:
```bash
source ml_env/bin/activate # if you are using Linux or macOS
# or
ml_env\Scripts\activate # if you are using Windows
```

- Installation of dependencies:
```bash
pip install -r requirements.txt
```

### c) Using uv

- Create a virtual environment (you can specify the Python version):
```bash
uv venv ml_env --python 3.13
```

- Activation:
```bash
source ml_env/bin/activate # Linux or macOS
# or
ml_env\Scripts\activate # Windows
```

- Installation of dependencies:
```bash
uv pip install -r requirements.txt
```

# Contributing
If you want to contribute to this project, follow these steps:

1. Clone the repository locally:
```bash
git clone https://github.com/ldan-dev/ml_env.git
```

2. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

3. Add your changes:
```bash
git add .
```

4. Commit your changes:
```bash
git commit -m "Add a descriptive commit message"
```

5. Push the branch to the remote repository:
```bash
git push origin feature/your-feature-name
```

6. Open your web browser, go to the repository on GitHub, and create a **Pull Request**.

# Use of the repository
This repository is for educational purposes only. Feel free to use and modify the code, but please give appropriate credit to the original author.
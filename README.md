TCC_MAIN_METHOD
=====

---
Pacotes utilizados
---

* [Scikit-Image](https://scikit-image.org/)
* [Scikit-Learn](https://scikit-learn.org/)
* [Numpy](https://docs.scipy.org/doc/numpy-1.13.0/reference/)
* [Scipy](https://www.scipy.org/)
* [Conda (opcional)](https://conda.io/en/latest/)

---
Como instalar?
---

Recomendo utilizar um gerenciador de ambiente virtual python, preferencialmente o Conda. Os passos a seguir consideram que você utilizou o Conda e um sistema Linux.

1. Crie um ambiente virtual (diga sim para todos os pedidos de instalação de pacotes)

    ```shell
    > conda create -n modeloPython python=3
    ```

2. Execute seu ambiente virtual

    ```shell
    > source activate modeloPython
    ```

3. Instale os pacotes necessários (ao usar os dois comandos abaixo, o conda já irá instalar uma série de outros pacotes necessários)

    ```shell
    > conda install scikit-image
    > conda install scikit-learn
    ```

---
Como executar?
---

1. Caso não exista, crie uma pasta chamada "saved_data" na raíz do projeto

2. Coloque a base de dados numa pasta chamada "Database", também na raíz do projeto. Dentro dela deve conter, para cada imagem, uma pasta contendo duas imagens, com os seguintes nomes: numeroDaImagem (imagem marcada) e numeroDaImage_orig (imagem original), SEM A EXTENSÂO DA IMAGEM. Exemplo:



    ```
    /Database
    --/1
    --/--/1
    --/--/1_orig
    --/2
    --/--/2
    --/--/2_orig

    etc...

    ```

3. Execute o arquivo runTests.py usando o comando

    ```shell
    > python runTests.py
    ```
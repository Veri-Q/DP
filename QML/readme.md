- Please note that due to the randomness of quantum machine learning model training, the values of $\kappa^*$ may differ from Table 4.
- EC_9, GC_9, AI_8, Mnist_10, Fashion_4 in Table 4

    1. Install `virtualenv` for creating virtual environments.
        ```bash
        pip install virtualenv
        ```

    2. Create a virtual environment `qml` and activate it
        ```bash
        virtualenv qml && source ./qml/bin/ activate
        ```

    3. Install required libraries in the virtual environment `qml`.
        ```bash
        pip install -r requirements.txt
        ```

        Run scripts
        ```bash
        python ec_9.py
        ```

        ```bash
        python gc_9.py
        ```

        ```bash
        python ai_9.py
        ```

        ```bash
        python mnist10.py
        ```

        ```bash
        python fashion4.py
        ```
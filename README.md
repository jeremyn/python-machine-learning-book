# Working Environment for *Python Machine Learning*

(Note: this repository is based on the first edition of *Python Machine Learning*, including the link to the book below. If you want to buy this book, I encourage you to go to the author's personal website, linked below, and from there find the most recent edition.)

This repository is a place for me to use and experiment with code from Sebastian Raschka's book *Python Machine Learning*. Any given code in here might be directly from the book, or entirely mine, or a mix.

I absolutely do not want to take credit for anything created by the book's author. Here are some links to learn more about the book, the author, and to see code for the book directly from the book's author:

* [Book at PacktPub](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning) (the book's publisher)

* [Home page for Sebastian Raschka](https://sebastianraschka.com) (the book's author)

* [Author's code for the book on GitHub](https://github.com/rasbt/python-machine-learning-book)

If you want to include or redistribute anything from this repository in your own project, please first check the book's code repository to see if you can get what you want from there so you can credit the book's author solely and directly.

## Requirements

~~The code in this repository should work with Python 3.5.1 and the dependencies in the included `requirements.txt`.~~

Security problems have been found in some of the original dependencies. I don't care to actively maintain this repository, so please consider this code read-only. I've renamed `requirements.txt` to `INSECURE.requirements` to make the problem clear while keeping the package list available.

## Datasets

Some code requires data files to run. These data files are not included in this repository. Please see Sebastian Raschka's book or code for instructions on where to get these files.

By default, the code expects these data files to be in a `datasets` directory (and `mnist` subdirectory for the MNIST data), but it should be straightforward to modify the code to read from other locations if you want.

## License

The book's author [has released his code under the MIT license](https://github.com/rasbt/python-machine-learning-book/blob/f07bacb9f678964ea0d79b2b0f8c66372b59ed77/LICENSE.txt), included here as `LICENSE.raschka.txt`. My own code and modifications are also released under the MIT license, included here as `LICENSE.txt`.

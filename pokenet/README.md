# pokenet
**CNN** that classifies 5 different pokemons from an image. The classes are _"bulbasaur", "charmander", "mewtwo", "pikachu"_ and _"squirtle"_. My version of https://github.com/grenlavus/smallvggnet. The data set is pretty small: ≈ _280 samples per class_. I use data augmentation to enhance the dataset. I have included some **Bath Normalization** and **Drop out** layers to combat overfitting. This version implements an _Early Stop callback_ if testing loss is below or equal to a threshold (th=0.19) during training. These are some results to test the net's generalization:

|        Input Image        |Number Recognition             |Output Matrix|
|---------------------------|-------------------------------|------------------|
|![sudoku01](https://user-images.githubusercontent.com/8327505/150041359-75bd20e6-3df4-469b-8ce0-d220e8bb6e30.png)|![sudoku02](https://user-images.githubusercontent.com/8327505/150041398-54cfdeea-dd14-4cc8-aba1-162b25511e10.png) |![sudokuMat](https://user-images.githubusercontent.com/8327505/150041406-ef4834b8-f396-4b62-85da-24c0848ee174.png) |


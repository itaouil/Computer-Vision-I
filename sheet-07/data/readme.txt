1. Data[1]: Each hand shape consists of 56 corresponding landmark points (x,y).
The training data consists of 39 shapes and the testing data consists of 1 shape.

2. hand_landmarks.txt contains landmark points (x,y).

3. Training Data: hands_aligned_train.txt contains relevant data. Each shape is
arranged as a column arranged as [x_0 x_1 ... x_55 y_0 y_1 y_55]^T. The header
of the file indicates the [numRows numCols] of the succeeding matrix.

4. Test Data: hands_aligned_test.txt holds relevant data and is arranged in the
same manner as the training data.

------------------------------------------------------------------------------
                               REFERENCES:
------------------------------------------------------------------------------
[1] http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=403
last accessed on 07th Dec 2015.

# aeternaSpeech 

This repository is an adaptation from the aeSpeech repository: https://github.com/jcvasquezc/AEspeech that extracts features from speech signals using pre-trained autoencoders. In our work we look to expand upon this and use the derived features to try and assess if a given speaker has either Parkinson's disease (PD) or cleft lip and palate (CLP).

Moreover, we look to compare different speech file representations (broadband/narrowband mel-spectrograms and wavelet-based scalograms) to see if a given representation is better for the classification task. Performance is (will be) evaluated based on the classification ability using each representation, as well as comparing tje sound quality from a given reconstructed speech file using a given representation (transformation from spectrogram to wav file done using DiffWave speech synthesizer).



The work is on-going, but currently one can derive the different representations from a directory of speech files (narrowband/broadband spectrograms and wavelet scalograms) by navigating to "/code/speechRepAnalysis/" and running "get_reps.py" (must specify path to a directory of speech files in command line)**. 

One can then train either a convolutional auto-encoder (CAE) or recurrent auto-encoder (RAE) by running "TrainCAE.py" or "TrainRAE.py" respectively, and specifying the number of bottle neck features desired (256 is standard), and the path to the speech file representations: "path/to/audios/../reps/'nb, bb, or wvlt'/train/" in the command line. Training/validation results can be visualized in the "view_recon.ipynb" Jupyter Notebook. Additionally, one can visualize the reconstructed signal representation in the same notebook.

The reconstruction error per frequency band for a dataset of PD and healthy control (HC) speech files** can be obtained by running "get_reconError.py". In the command line, one must specify the path to the directory of PD/HC speech files ("code/speechRepAnalysis/pdSpanish/speech/").
In the "view_recon.ipynb" Jupyter Notebook, one can see the average reconstruction error per frequency band for both PD vs HC speakers. The results hypothetically should show that the average error is significant making it a viable feature for distinguishing between the two classes. This is also seen via the Wilcoxon score and p-value also reported in the plots.

Running one of "pdnnEval.py", "pdnnEvalAgg.py", "pdsvmEval.py", or "pdsvmEvalAgg.py" will first (if not done already) derive features (both bottle-neck features and reconstruction error) from the database of PD/HC speech files using the learned parameters from the previously trained auto-encoders. With the features, classification (using either a deep neural net (dnn) or support vector machine (svm)) is then performed using 10-fold cross validation. 

The pdnnEval.py and pdsvmEval.py scripts allow for the specification of which DDK utterance one would like to consider. The 'Agg' files utilize all DDK utterances. Performance so far found to be better when a classifer is trained individually for each DDK utterance type. 

For the dnn-based classifers, training/testing is done on a single speaker at a time. For the svm-based classifiers, PCA is used to reduce the dimensionality of the training set.

Results can be visulalized in either the "view_classRes_dnn.ipynb" or "view_classRes_svm.ipynb" notebooks.  



NOTE: PARAMETERS FOR ALL processes can be specified in params.config (e.g. speech representation parameters - mel spectrogram window/shift length, AE architecture parameters, classifier parameters).

*Note regarding database used for training AEs: we utilized the "TEDx Spanish Corpus" from https://www.openslr.org/67/. The database contains over 24 hours of speech from TEDx talks all given in Spanish, with most of the speakers being men.

**Note regarding PD database: 100 (Colombian) Spanish speakers from the PC-GITA corpus [1] (50 PD, 50 HC). Each administered diadochokinetic (DDK) test. Six different DDK utternace types: pataka, pakata, petaka, tatata, papapa, kakaka. 

[1] J.R. Orozco-Arroyave, J.D. Arias-Londoño, J.F. Vargas-Bonilla, M.C. Gonzalez Rátiva, E. Nöth, “New Spanish Speech Corpus Database for the Analysis of People Suffering from Parkinson’s Disease”, in: Proceedings of the 9th LREC, ERLA, 2014, pp. 342–347
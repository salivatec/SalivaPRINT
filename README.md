# SalivaPRINT

SalivaPRINT available commands can be checked anytime by using -h as argument. The following commands are currently implemented (version 0.1.1):

-v: Displays the program and required libraries version.

-h: Displays the help menu. Lists the available commands.

-build outputfile: Builds a new molecular feature matrix from Experion™ output files using config.cfg as the configurations file.

-view inputfile: Shows a visual representation of the dataset previously built using the –build flag

-learn inputfile outputfile: Builds a classifier from inputfile dataset. Uses the name given as outputfile for saving the created classifier.

-classify classifier_file dataset: Classifies the dataset using the previously trained classifier.

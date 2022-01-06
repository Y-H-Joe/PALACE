Reaction Template Generator

Introduction
------------
The Reaection Template Generator generates reaction templates in the format 
required by the kinetic model generation tool Genesys.

License
-------

"Reaction Template Generator" Copyright 2017 Laboratory for Chemical Technology
Technologiepark 914
9052 Zwijnaarde - Belgium
Author: Pieter Plehiers
email:  pieter.plehiers@ugent.be

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
Runnable jar file and batch file can be found in this folder

USE
---

Both the source code as a compiled .jar file of the main algorithm are provided.

To run the compiled version following must be observed:
-	RDT1.5.jar is located in the same directory as the compiled version of the code.
	It is important that the RDT program is named "RDT1.5.jar". It may be possible to
	use other versions of RDT, but this may impact the functioning of the code. For
	convenience RDT1.5.jar is also provided with this code.
-	Different input formats are possible: Text file (structured according to InputTemGen.txt),
	a single .rxn file, a directory containing several .rxn files, or a chemkin input file
-	Make sure java is on your file path in the environment variables
-	There is no directory in the working directory with the same name as the one you want
	the program to write the output too (unless running with the command -oC)
	
Commands for running the compiled .jar file:
-	Run:
	java -Xmx1g -Dlog4j.configuration="file:log4j.properties" -jar [Compiled File Name].jar -i [Input File or Directory Name] -o [Reaction Templates File Name]>[Log File Name]
	alternatively:
	"[Absolute location of java.exe]" -Dlog4j.configuration="file:log4j.properties" -jar [Compiled File Name].jar -i [Input File or Directory Name] -o [Reaction Templates File Name]>[Log File Name]
-	Additional options/commands:
	java -Xmx1g -Dlog4j.configuration="file:log4j.properties" -jar [Compiled File Name].jar -h
	-oD: Name of the output directory, to which all reaction directories will be written.
	-oC: If specified, the existence of the demanded output directory will not be verified. Risk of losing data.
	-zK: Will generate an arrhenius block for the kinetics with all parameters set to 0.
	-nC: No product constraints on heavy atom count will be added.
	-s:  Increase specificity by adding connected hetero atoms to the reaction center.
	-h:  Help page for command line usage.

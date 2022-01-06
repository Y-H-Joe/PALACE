@ECHO OFF
IF /i {%1}=={ECHO} ECHO ON&SHIFT

del logfile
ECHO FileGen started
java -Xmx1g -Dlog4j.configuration="file:log4j.properties" -jar ReactionTemplateGenerator.jar -i Input -o reactionFamilies.xml -oD ReactionTemplates > logfile
ECHO FileGen ended
PAUSE


package io;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import javax.xml.XMLConstants;
import javax.xml.transform.Source;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;
import javax.xml.validation.Validator;

import org.xml.sax.SAXException;

import org.apache.log4j.Logger;

public class XMLValidator {

	private File xmlFile;
	private Schema xmlSchema;
	private static Logger logger	=	Logger.getLogger(XMLValidator.class);
	
	public XMLValidator(File xml, URL schema){
		this.xmlFile				=	xml;
		SchemaFactory schemaFact	=	SchemaFactory.newInstance(XMLConstants.W3C_XML_SCHEMA_NS_URI);
		this.xmlSchema				=	null;
	
		try{
			this.xmlSchema	=	schemaFact.newSchema(schema);
		}
		catch(SAXException e){}
	}
	
	public void validate(){
		
		Source xmlSource=	new StreamSource(xmlFile);
		Validator val	=	xmlSchema.newValidator();
		
		try{
			val.validate(xmlSource);
			logger.info("Network data xml file " + xmlFile.getName() + " is valid.");
		}
		catch(SAXException e){
			logger.fatal("Network data xml file " + xmlFile.getName() + " is not valid.");
			logger.fatal("Invalid because: " + e.getLocalizedMessage());
			System.exit(-1);
		}
		catch(IOException e){}
	}
}

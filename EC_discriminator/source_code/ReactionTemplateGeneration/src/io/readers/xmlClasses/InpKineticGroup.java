//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2017.02.21 at 02:50:12 PM CET 
//


package io.readers.xmlClasses;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for inp-kinetic-group complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="inp-kinetic-group">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="smarts" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="DA" use="required" type="{http://www.w3.org/2001/XMLSchema}double" />
 *       &lt;attribute name="DEa" use="required" type="{http://www.w3.org/2001/XMLSchema}double" />
 *       &lt;attribute name="DN" type="{http://www.w3.org/2001/XMLSchema}double" />
 *       &lt;attribute name="comments" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "inp-kinetic-group")
public class InpKineticGroup {

    @XmlAttribute(name = "smarts", required = true)
    protected String smarts;
    @XmlAttribute(name = "DA", required = true)
    protected double da;
    @XmlAttribute(name = "DEa", required = true)
    protected double dEa;
    @XmlAttribute(name = "DN")
    protected Double dn;
    @XmlAttribute(name = "comments")
    protected String comments;

    /**
     * Gets the value of the smarts property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSmarts() {
        return smarts;
    }

    /**
     * Sets the value of the smarts property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSmarts(String value) {
        this.smarts = value;
    }

    /**
     * Gets the value of the da property.
     * 
     */
    public double getDA() {
        return da;
    }

    /**
     * Sets the value of the da property.
     * 
     */
    public void setDA(double value) {
        this.da = value;
    }

    /**
     * Gets the value of the dEa property.
     * 
     */
    public double getDEa() {
        return dEa;
    }

    /**
     * Sets the value of the dEa property.
     * 
     */
    public void setDEa(double value) {
        this.dEa = value;
    }

    /**
     * Gets the value of the dn property.
     * 
     * @return
     *     possible object is
     *     {@link Double }
     *     
     */
    public Double getDN() {
    	//TODO:Good default?
    	if(dn == null)
    		return 0.0;
    	
        return dn;
    }

    /**
     * Sets the value of the dn property.
     * 
     * @param value
     *     allowed object is
     *     {@link Double }
     *     
     */
    public void setDN(Double value) {
        this.dn = value;
    }

    /**
     * Gets the value of the comments property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getComments() {
        return comments;
    }

    /**
     * Sets the value of the comments property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setComments(String value) {
        this.comments = value;
    }

}

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
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for inp-substituent complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="inp-substituent">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="inp-chiral-center" type="{}inp-chiral-center"/>
 *       &lt;/sequence>
 *       &lt;attribute name="smarts" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "inp-substituent", propOrder = {
    "inpChiralCenter"
})
public class InpSubstituent {

    @XmlElement(name = "inp-chiral-center", required = true)
    protected InpChiralCenter inpChiralCenter;
    @XmlAttribute(name = "smarts", required = true)
    protected String smarts;

    /**
     * Gets the value of the inpChiralCenter property.
     * 
     * @return
     *     possible object is
     *     {@link InpChiralCenter }
     *     
     */
    public InpChiralCenter getInpChiralCenter() {
        return inpChiralCenter;
    }

    /**
     * Sets the value of the inpChiralCenter property.
     * 
     * @param value
     *     allowed object is
     *     {@link InpChiralCenter }
     *     
     */
    public void setInpChiralCenter(InpChiralCenter value) {
        this.inpChiralCenter = value;
    }

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

}

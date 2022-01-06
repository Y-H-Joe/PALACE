//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2017.02.21 at 02:50:12 PM CET 
//


package io.readers.xmlClasses;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for config complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="config">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="inp-temperature" type="{http://www.w3.org/2001/XMLSchema}double" minOccurs="0"/>
 *         &lt;element name="inp-reaction-family" type="{}inp-reaction-family" maxOccurs="unbounded"/>
 *         &lt;element name="inp-product-constraints" type="{}inp-constraint" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlRootElement
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "config", propOrder = {
    "inpTemperature",
    "inpReactionFamily",
    "inpProductConstraints"
})
public class Config {

    @XmlElement(name = "inp-temperature")
    protected Double inpTemperature;
    @XmlElement(name = "inp-reaction-family", required = true)
    protected List<InpReactionFamily> inpReactionFamily;
    @XmlElement(name = "inp-product-constraints")
    protected List<InpConstraint> inpProductConstraints;

    /**
     * Gets the value of the inpTemperature property.
     * 
     * @return
     *     possible object is
     *     {@link Double }
     *     
     */
    public Double getInpTemperature() {
        return inpTemperature;
    }

    /**
     * Sets the value of the inpTemperature property.
     * 
     * @param value
     *     allowed object is
     *     {@link Double }
     *     
     */
    public void setInpTemperature(Double value) {
        this.inpTemperature = value;
    }

    /**
     * Gets the value of the inpReactionFamily property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the inpReactionFamily property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getInpReactionFamily().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link InpReactionFamily }
     * 
     * 
     */
    public List<InpReactionFamily> getInpReactionFamily() {
        if (inpReactionFamily == null) {
            inpReactionFamily = new ArrayList<InpReactionFamily>();
        }
        return this.inpReactionFamily;
    }

    /**
     * Gets the value of the inpProductConstraints property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the inpProductConstraints property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getInpProductConstraints().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link InpConstraint }
     * 
     * 
     */
    public List<InpConstraint> getInpProductConstraints() {
        if (inpProductConstraints == null) {
            inpProductConstraints = new ArrayList<InpConstraint>();
        }
        return this.inpProductConstraints;
    }
    
    public void addInpProductConstraint(InpConstraint constraint){
    	if (inpProductConstraints == null) {
            inpProductConstraints = new ArrayList<InpConstraint>();
        }
        this.inpProductConstraints.add(constraint);
    }
    
    /**Add a reaction family to the list of families. If no list exists yet, create one
     * 
     * @param reactionFamily
     */
    public void addInpReactionFamily(InpReactionFamily reactionFamily){
    	
    	if(inpReactionFamily == null)
    		inpReactionFamily	=	new ArrayList<InpReactionFamily>();
    	
    	inpReactionFamily.add(reactionFamily);
    }

}

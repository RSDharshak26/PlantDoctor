'use client'

import React, { useState } from "react";

export default function Page(){

    // state declaration for storing user image uploads
    const [selectedfile,setselectedfile] = useState(null);

    const handleSubmit = async(e) => {
        e.preventDefault() // stops the browser from doing its default action (reloading/navigating).
        console.log("Preparing to send data ")
        
        //following is to access the file stored in DOM directly
        const fileinput = e.currentTarget.elements.namedItem('file_upload')
        const image = fileinput.files[0]
        
        // appending the file to the form that will be sent to backend
        const formdata = new FormData();
        formdata.append('file',image)
        const response = await fetch("http://localhost:6000/inference",{
            method : "POST",
            body : formdata
        });
        console.log("POST request sent")
    };
    


return (
    <form onSubmit={handleSubmit}>

    <h2>Please insert your image. ONLY image please </h2>
    <input type = "file" name = "file_upload" accept = "image/*"/>
    <button type="submit">Upload Image</button>
    </form>
);

}
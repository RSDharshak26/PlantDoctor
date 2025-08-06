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
        
        // API URL - will be updated after SAM deployment
        const apiUrl = "https://5zmm7bjmtnhz5mxqnkini5kyvm0npiue.lambda-url.us-east-1.on.aws/"
        const response = await fetch(`${apiUrl}/receive`,{
            method : "POST",
            body : formdata
        });
        console.log("POST request sent")
    };
    

return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-green-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
                            <span className="text-white text-xl font-bold">🌱</span>
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900">Plant Doctor</h1>
                    </div>
                    <p className="text-green-600 font-medium">AI-Powered Plant Disease Detection</p>
                </div>
            </div>
        </header>

        {/* Main Content */}
        <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            {/* Hero Section */}
            <div className="text-center mb-12">
                <h2 className="text-4xl font-bold text-gray-900 mb-4">
                    Diagnose Your Plants with AI
                </h2>
                <p className="text-xl text-gray-600 mb-8">
                    Upload a photo of your plant and get instant disease diagnosis and treatment recommendations
                </p>
                <div className="flex justify-center space-x-4 text-sm text-green-600">
                    <div className="flex items-center">
                        <span className="mr-2">✅</span>
                        Instant Analysis
                    </div>
                    <div className="flex items-center">
                        <span className="mr-2">✅</span>
                        Accurate Results
                    </div>
                    <div className="flex items-center">
                        <span className="mr-2">✅</span>
                        Treatment Tips
                    </div>
                </div>
            </div>

            {/* Upload Section */}
            <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="text-center">
                        <label htmlFor="file_upload" className="cursor-pointer">
                            <div className="border-2 border-dashed border-green-300 rounded-xl p-8 hover:border-green-400 transition-colors">
                                <div className="text-6xl mb-4">📸</div>
                                <p className="text-lg font-medium text-gray-700 mb-2">
                                    Click to upload plant image
                                </p>
                                <p className="text-sm text-gray-500">
                                    Supports JPG, PNG, GIF up to 10MB
                                </p>
                            </div>
                        </label>
                        <input 
                            type="file" 
                            name="file_upload" 
                            accept="image/*"
                            id="file_upload"
                            className="hidden"
                        />
                    </div>

                    <div className="text-center">
                        <button 
                            type="submit"
                            className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors duration-200 flex items-center justify-center mx-auto"
                        >
                            🔍 Analyze Plant
                        </button>
                    </div>
                </form>
            </div>

            {/* Features Section */}
            <div className="grid md:grid-cols-3 gap-6 mt-12">
                <div className="bg-white rounded-xl p-6 shadow-lg">
                    <div className="text-3xl mb-3">🤖</div>
                    <h3 className="font-semibold text-gray-900 mb-2">AI-Powered</h3>
                    <p className="text-gray-600">Advanced machine learning algorithms for accurate plant disease detection</p>
                </div>
                <div className="bg-white rounded-xl p-6 shadow-lg">
                    <div className="text-3xl mb-3">⚡</div>
                    <h3 className="font-semibold text-gray-900 mb-2">Instant Results</h3>
                    <p className="text-gray-600">Get your plant diagnosis in seconds, not hours or days</p>
                </div>
                <div className="bg-white rounded-xl p-6 shadow-lg">
                    <div className="text-3xl mb-3">💡</div>
                    <h3 className="font-semibold text-gray-900 mb-2">Expert Tips</h3>
                    <p className="text-gray-600">Receive personalized treatment recommendations for your plants</p>
                </div>
            </div>
        </main>

        {/* Footer */}
        <footer className="bg-green-800 text-white py-8 mt-16">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                <p className="text-green-100">
                    © 2024 Plant Doctor. Helping you keep your plants healthy with AI technology.
                </p>
            </div>
        </footer>
    </div>
);

}
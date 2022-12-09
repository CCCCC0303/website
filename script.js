function display(event)
 {
  let input_image = document.getElementById("input_image")
  input_image.src = URL.createObjectURL(event.target.files[0]);
  document.getElementById("input_image_container").style.display = "block";
 }
    
//Predict emotion and display output
async function predict_classification()
 {
  let input = document.getElementById("input_image");
  //Preprocessing steps 
  //Resize to 416*416
  let step1 = tf.browser.fromPixels(input).resizeNearestNeighbor([416,416]).expandDims().div(255.0)
        //load model into js
  const model = await tf.loadLayersModel('https://raw.githubusercontent.com/Timyangcn/wandan/main/model.json');
  cc = model.summary()
  pred = model.predict(step1)
  pred.print()
  //This array is encoded with index i = corresponding classification. In dataset, 0 = qualified, 1 = failed, 2 = feature
  classification = ["是", "否", "是(且有磨損點)"]
  //At which index in tensor we get the largest value ?
  pred.data()
      .then((data) => {console.log(data)
       output = document.getElementById("output_chart")
       output.innerHTML = ""
       max_val = -1
       max_val_index = -1
    for(let i=0;i<data.length;i++)
    {
     style_text = "width: "+data[i]*150+"px; height: 25px; position:relative; margin-top: 3vh; background-color: violet; "
     output.innerHTML+="<div style = '" +style_text+ "'></div>"
     if(data[i] > max_val)
     {
      max_val = data[i]
      max_val_index = i
     }
    }
    CLASSIFICATION_DETECTED = classification[max_val_index]
    document.getElementsByClassName("output_screen")[0].style.display="flex";
    document.getElementById("output_text").innerHTML=""
    document.getElementById("output_text").innerHTML = "<p>是否通過標準: " + CLASSIFICATION_DETECTED + "(" + (max_val*100).toFixed(2) + "% probability)</p>"
  }) 

 }

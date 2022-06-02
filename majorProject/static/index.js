let input=document.getElementById("titlebody").value;
function getInput(){
input=document.getElementById("titlebody").value;
console.log(input);
}
function predict(){
console.log(input)
if(input.length===0){
alert("Input can't be empty")
}
else{
if(onlyNumbers(input)){
alert("Input Can't be number");
}
else{
let request= new XMLHttpRequest () ;
request.open("post", "https://127.0.0.1:9000/predict");
request.setRequestHeader("Content-type", "application/json");
request.send(input);
request.onload = () => {
console.log(request);
}
}
}

}
function onlyNumbers(str) {
  return /^[0-9]+$/.test(str);
}



let title =window.location.href.split('/');
title=title[title.length-1];
switch(title){
case "titleModule": document.getElementById('title').innerHTML= "Using Title";
        break;
case "bodyModule":document.getElementById('title').innerHTML= "Using body";
        break;
case "bodyAndTitleModule":document.getElementById('title').innerHTML= "Using body and title"
        break;
}

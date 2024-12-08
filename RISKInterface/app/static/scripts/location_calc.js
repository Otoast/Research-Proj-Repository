let map, marker;
// Put your own gmap api key here
let gmap_api_key = null;
async function initMap(lat, lng) {
    (g=>{var h,a,k,p="The Google Maps JavaScript API",c="google",l="importLibrary",q="__ib__",m=document,b=window;b=b[c]||(b[c]={});var d=b.maps||(b.maps={}),r=new Set,e=new URLSearchParams,u=()=>h||(h=new Promise(async(f,n)=>{await (a=m.createElement("script"));e.set("libraries",[...r]+"");for(k in g)e.set(k.replace(/[A-Z]/g,t=>"_"+t[0].toLowerCase()),g[k]);e.set("callback",c+".maps."+q);a.src=`https://maps.${c}apis.com/maps/api/js?`+e;d[q]=f;a.onerror=()=>h=n(Error(p+" could not load."));a.nonce=m.querySelector("script[nonce]")?.nonce||"";m.head.append(a)}));d[l]?console.warn(p+" only loads once. Ignoring:",g):d[l]=(f,...n)=>r.add(f)&&u().then(()=>d[l](f,...n))})({
      key: gmap_api_key,
      v: "weekly",
    });
    
    const { Map } = await google.maps.importLibrary("maps");
    map = new Map(document.getElementById("map"), {
      center: { lat: lat, lng: lng},
      zoom: 8,
    });
    marker = new google.maps.Marker({
      position: { lat: lat, lng: lng}, // Initial marker position
      map,
    });
    if (typeof google !== 'object' && typeof google.maps !== 'object') {
      document.getElementById("map").innerHTML = "Are you using an adblocker? Please disable the extension tempoarially for Google Maps to function."
    }
    // map.addListener('center_changed', function () {
    //     // Get the new center of the map
    //     var center = map.getCenter();
    //     console.log("New center:", center.lat(), center.lng());
    // });
    
}
  
function changeMapLocation(lat, lng, location){ 
    map.setCenter({lat, lng});
    marker.setPosition({lat, lng});
    document.getElementById('estimated-location').innerHTML = location;
}

initMap(30.601433, -96.314464);


function checkNumber(event) {
    const key = event.key;
    if (!(/^\d$/.test(key) || key === '-')) event.preventDefault();  
}
  
  
  function getCookie(name) {
    let cookieValue;
    if (document.cookie && document.cookie !== '') {
        let cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            let cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }        
    }
    return cookieValue
}
function output_response(data, is_location=true) {
    p = document.getElementById('weather-cond-paragraph').innerHTML = data['content'];
    
    document.getElementById('recc-paragraph').innerHTML = data['safety_recc'];
    
    stoplight_image = document.getElementById('stoplight-image');
    stoplight_text = document.getElementById('risk-level');
    
    risk_level = data['risk'];
    url = "../images/"
    if (risk_level === 'Low') {url += "green_light.png";}
    else if (risk_level == 'Moderate') {url += 'yellow_light.png';}
    else {url += 'red_light.png';}
    stoplight_image.style.backgroundImage = `url(static/images/${url})`;
    stoplight_text.innerHTML = "Risk Level:<br/>" + data['risk'];  
    if (is_location) {
        document.getElementById('lat').value = data['lat'];
        document.getElementById('lng').value = data['lng'];
    };
      
    changeMapLocation(data['lat'], data['lng'], data['location']);
}


  
function submit_form(lat='NA', lng='NA', zip='NA', geo_clicked=false) {
    is_location = document.getElementById('lat-lon-btn').classList.contains("parameter_active");  
    if ( !geo_clicked && ( ((lat === 'NA' || lng === 'NA') && is_location) || (!is_location && zip === 'NA')  ) ) {
        alert("Parameters empty!");
        return -1;
    }
    let data = {'lat' : lat, 'lng' : lng, 'zip' : zip, 'type':'location'};
    fetch('/submit', {
        method : 'POST',
        body : JSON.stringify(data),
        headers : {'X-CSRFToken' : getCookie('csrftoken')}
    })
    .then((response) => response.json())
    .then((data) => {
        if (Object.keys(data).includes('message')) {
            alert('Error in processing input. Please try again!');
            return;
        }
        is_location = document.getElementById('lat-lon-btn').classList.contains("parameter_active");   
        output_response(data, is_location);
    })


}


document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll(".output-select").forEach(selector => {

        selector.addEventListener('click', function () {
          arrow = selector.getElementsByClassName('dropdown-arrow')[0];
          paragraph = selector.parentElement.getElementsByClassName('output-paragraph')[0];
          
          paragraph.classList.toggle('visible');
          arrow.classList.toggle('rotate');
        })
      })
    
    document.getElementsByClassName('submit-btn')[0].addEventListener('click', function(event){
        event.preventDefault();
        is_location = document.getElementById('lat-lon-btn').classList.contains("parameter_active");   
        let lat, lng, zip;
        if (is_location) {
            lat = document.getElementById('lat').value;
            lng = document.getElementById('lng').value;
            submit_form(lat, lng);

        }     
        else {
            zip = document.getElementById('zip').value;
            submit_form('NA', 'NA', zip);
            
        }

    })

    var lat_lon_btn = document.getElementById('lat-lon-btn');
    lat_lon_btn.classList.add("parameter_active");
    lat_lon_btn.addEventListener('click', function(){
        document.querySelectorAll('.lat-lon-entry').forEach(element => {element.style.display = 'table-row';});
        document.querySelectorAll('.zip-entry').forEach(element => {element.style.display = 'none';});
        lat_lon_btn.classList.add("parameter_active");
        document.getElementById('zip-btn').classList.remove("parameter_active");
    })
    var zip_btn = document.getElementById('zip-btn');
    zip_btn.addEventListener('click', function(){
        document.querySelectorAll('.zip-entry').forEach(element => {element.style.display = 'table-row';});
        document.querySelectorAll('.lat-lon-entry').forEach(element => {element.style.display = 'none';});
        
        zip_btn.classList.add("parameter_active");
        document.getElementById('lat-lon-btn').classList.remove("parameter_active");
    })

    var geolocation_btn = document.getElementsByClassName('geolocation-btn')[0];
    geolocation_btn.addEventListener('click', function(event){
        document.getElementById('risk-level').innerHTML = 'Loading...';
        navigator.geolocation.getCurrentPosition((location_info) => {
            const latitude = location_info.coords.latitude;
            const longitude = location_info.coords.longitude;
            submit_form(latitude, longitude, 'NA', true);

    
         })
    })

})


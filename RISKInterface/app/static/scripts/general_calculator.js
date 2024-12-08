function checkNumber(event) {
  const key = event.key;
  if (!(/^\d$/.test(key) || key === '-')) event.preventDefault();  
}


function getCookie(name) {
  let cookieValue = null;
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

function output_response(data) {
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

}

function submit_RISK_form() {
  let form = document.getElementById('RISK-interface');
  let formData = new FormData(form);
  formData.append('type', 'general');
  let csrfToken = getCookie('csrftoken');

  fetch('/submit', {
      method: 'POST', 
      body : formData,
      headers : {
          'X-CSRFToken' : csrfToken,
      }
  })
  .then(response => response.json())
  .then(data => output_response(data));
}



document.addEventListener('DOMContentLoaded', function() {

  document.getElementById('basic-btn').addEventListener('click', function() {
    window.location.href = "basic-calculator"
  })
  document.getElementById('loc-btn').addEventListener('click', function() {
    window.location.href = "location-calculator"
  })

  document.getElementById('RISK-interface').addEventListener('submit', function(event) {
    event.preventDefault();
    submit_RISK_form(); 
  })
  document.querySelectorAll(".output-select").forEach(selector => {

    selector.addEventListener('click', function () {
      arrow = selector.getElementsByClassName('dropdown-arrow')[0];
      paragraph = selector.parentElement.getElementsByClassName('output-paragraph')[0];
      
      paragraph.classList.toggle('visible');
      arrow.classList.toggle('rotate');
    })
  })

})




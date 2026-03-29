window.onload = function() {
  let pages = ["page1","page2","page3","page4","page5","page6","page7"];

  function showPage(index){
    pages.forEach(p => {
      let el = document.getElementById(p);
      if(el) el.style.display = "none";
    });
    let currentEl = document.getElementById(pages[index]);
    if(currentEl) currentEl.style.display = "block";
  }

  showPage(0);

  // Page 1 → Page 2
  document.getElementById("next1").onclick = ()=>showPage(1);

  // Page2 → clg answer
  let correctDiv = document.createElement("div");
  correctDiv.className = "center-text-overlay";
  correctDiv.innerText = "Correct baby 💖";
  document.body.appendChild(correctDiv);

  document.getElementById("submit1").onclick = ()=>{
    let ans = document.getElementById("answer1").value.toLowerCase();
    if(ans==="clg"){ 
        correctDiv.style.display = "block"; 
        setTimeout(()=>{ correctDiv.style.display = "none"; showPage(2); }, 1500);
    } else alert("Try again 💖");
  };

  // Page3 → cards click
  document.querySelectorAll(".card").forEach(card=>{
    card.onclick = ()=>{ 
      let photo = card.getAttribute("data-photo"); 
      window.open(photo, "_blank"); 
    }
  });
  document.getElementById("next3").onclick = ()=>showPage(3);

  // Page4 → next
  document.getElementById("next4").onclick = ()=>showPage(4);

  // Page5 → "Do you remember this day?"
  document.getElementById("yes").onclick = ()=>{
    correctDiv.innerText = "I love you babe 😘"; 
    correctDiv.style.display = "block"; 
    setTimeout(()=>{ correctDiv.style.display = "none"; showPage(5); }, 1500);
  };
  document.getElementById("no").onclick = ()=>alert("Babe 😘");

  // Page6 → "Will you marry me?"
  document.getElementById("yes2").onclick = ()=>{
    correctDiv.innerHTML = "You said yes 💍<br>From this moment ..it's you and me forever ♾️<br>I promise you ❤️<br><img src='images/hug_bubu.jpg' style='width:150px;margin-top:15px'>"; 
    correctDiv.style.display = "block"; 
    setTimeout(()=>{ correctDiv.style.display = "none"; showPage(6); }, 2000);
  };
  document.getElementById("no2").onclick = ()=>alert("Babe 😢");
};
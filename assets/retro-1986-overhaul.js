(()=>{
const boot=()=>{
 const deck=document.querySelector('.wow-deck'); if(!deck||deck.dataset.boot1986)return; deck.dataset.boot1986='1';
 const overlay=document.createElement('div'); overlay.className='retro-boot'; overlay.innerHTML='<div class="retro-boot-box"><div>LOAN INTELLIGENCE BIOS v1.986</div><div class="retro-boot-line">MEMORY CHECK ........ 640K OK</div><div class="retro-boot-line">MODEM ............... CONNECTED</div><div class="retro-boot-line">LOADING XGBOOST .... <span>READY</span></div><div class="retro-boot-line">WARNING ............. USER HAS 92.87% ACCURACY</div><div class="retro-boot-prompt">PRESS ENTER TO CONTINUE_</div></div>';
 deck.appendChild(overlay);
 const style=document.createElement('style'); style.textContent='.retro-boot{position:absolute;inset:0;z-index:100;background:#10150d;color:#b7ff75;display:grid;place-items:center;font:18px VT323,monospace}.retro-boot-box{width:min(760px,80vw);border:3px solid #7d9270;padding:28px;box-shadow:8px 8px 0 #000;background:#11170e}.retro-boot-line{margin-top:12px}.retro-boot-line span{color:#ffb84d}.retro-boot-prompt{margin-top:28px;color:#ffb84d;animation:blink1986 .8s steps(2) infinite}@keyframes blink1986{50%{opacity:0}}.retro-boot.done{animation:bootOut .35s steps(3) forwards}@keyframes bootOut{to{opacity:0;visibility:hidden}}'; document.head.appendChild(style);
 const finish=()=>{overlay.classList.add('done');setTimeout(()=>overlay.remove(),400)};
 overlay.addEventListener('click',finish); document.addEventListener('keydown',function e(ev){if(!deck.classList.contains('open'))return;if(ev.key==='Enter'){finish();document.removeEventListener('keydown',e)}});
 setTimeout(finish,1800);
};
new MutationObserver(boot).observe(document.body,{childList:true,subtree:true});setTimeout(boot,250);
})();

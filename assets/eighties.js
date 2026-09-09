(() => {
  const boot = () => {
    document.body.classList.add('eighties-mode');

    const overlay = document.createElement('div');
    overlay.className = 'crt-overlay';
    overlay.innerHTML = '<div class="crt-scanlines"></div><div class="crt-noise"></div><div class="crt-corner crt-tl">REC ●</div><div class="crt-corner crt-tr">NTI-86</div><div class="crt-corner crt-bl">LOAN INTELLIGENCE</div><div class="crt-corner crt-br">VHS MODE</div>';
    document.body.appendChild(overlay);

    const badge = document.createElement('button');
    badge.className = 'eighties-badge';
    badge.type = 'button';
    badge.innerHTML = '☻ 1986 MODE';
    badge.title = 'Toggle the ridiculous 80s theme';
    badge.addEventListener('click', () => document.body.classList.toggle('eighties-mode'));
    document.body.appendChild(badge);

    const fs = document.createElement('button');
    fs.className = 'retro-fullscreen';
    fs.type = 'button';
    fs.innerHTML = '▣ FULL SCREEN';
    fs.addEventListener('click', async () => {
      try {
        if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
        else await document.exitFullscreen();
      } catch (_) {}
    });
    document.body.appendChild(fs);

    document.querySelectorAll('.primary, .ghost, .text-btn').forEach((b) => {
      if (!b.dataset.retroBound) {
        b.dataset.retroBound = '1';
        b.addEventListener('mouseenter', () => { b.style.setProperty('--wiggle', '1'); });
      }
    });

    // Tiny fake-terminal boot sequence: decorative only, never presented as model output.
    const hero = document.querySelector('.hero-copy');
    if (hero && !hero.querySelector('.retro-terminal')) {
      const term = document.createElement('div');
      term.className = 'retro-terminal';
      term.innerHTML = '<span>LOAN_OS v1.0</span><br>INSERT APPLICANT...<br>LOADING XGBOOST... <b>OK</b><br>PRESS START TO PREDICT <i>_</i>';
      hero.appendChild(term);
    }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();

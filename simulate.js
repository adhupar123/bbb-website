/* simulate.js
  - Robust TSV loader & header matching
  - OLS regression (trained on rows with experimental logBB)
  - Sliders update instantly when dropdown changes
  - Two-compartment RK4 simulation for both predicted & experimental LogBB
  - Plotly visualization
*/

(async function(){
  // --- helpers ---
  function toNum(x){ return (x === undefined || x === '') ? NaN : +x; }

  async function fetchText(path){
    const r = await fetch(path);
    if(!r.ok) throw new Error('Failed to fetch ' + path + ' (' + r.status + ')');
    return r.text();
  }

  function parseTSV(txt){
    const lines = txt.trim().split(/\r?\n/).filter(Boolean);
    const hdr = lines[0].split('\t').map(h => h.trim());
    const rows = lines.slice(1).map(l => {
      const cols = l.split('\t');
      const obj = {};
      hdr.forEach((h,i) => obj[h] = (i < cols.length) ? cols[i] : '');
      return obj;
    });
    return { hdr, rows };
  }

  // Matrix utilities for OLS
  function transpose(A){ return A[0].map((_,i)=> A.map(r=> r[i])); }

  function matMul(A,B){
    const m=A.length, n=A[0].length, p=B[0].length;
    const C = Array.from({length:m}, ()=> Array(p).fill(0));
    for(let i=0;i<m;i++){
      for(let k=0;k<n;k++){
        const a=A[i][k];
        for(let j=0;j<p;j++) C[i][j] += a * B[k][j];
      }
    }
    return C;
  }

  function invert(M){
    const n=M.length;
    const A=M.map(r=> r.slice());
    const I=Array.from({length:n}, (_,i)=> Array.from({length:n}, (__,j)=> i===j?1:0));
    for(let i=0;i<n;i++){
      if(Math.abs(A[i][i]) < 1e-12){
        let swap = i+1;
        while(swap<n && Math.abs(A[swap][i])<1e-12) swap++;
        if(swap===n) throw new Error('Singular matrix in inversion');
        [A[i], A[swap]] = [A[swap], A[i]];
        [I[i], I[swap]] = [I[swap], I[i]];
      }
      const piv = A[i][i];
      for(let j=0;j<n;j++){ A[i][j] /= piv; I[i][j] /= piv; }
      for(let r=0;r<n;r++){
        if(r===i) continue;
        const f = A[r][i];
        for(let c=0;c<n;c++){ A[r][c] -= f * A[i][c]; I[r][c] -= f * I[i][c]; }
      }
    }
    return I;
  }

  function ols(X, y){
    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    const XtXinv = invert(XtX);
    const Xty = matMul(Xt, y.map(v=>[v]));
    const betaMat = matMul(XtXinv, Xty);
    return betaMat.map(r=> r[0]);
  }

  // RK4 two-compartment simulate
  function simulateTwoComp(logBB, P_eff, Tmax, dt){
    const K = Math.pow(10, logBB);
    const steps = Math.max(2, Math.floor(Tmax/dt)+1);
    const t = new Array(steps);
    const Cb = new Array(steps);
    const Cbr = new Array(steps);
    let cb = 1.0, cbr = 0.0;
    t[0]=0; Cb[0]=cb; Cbr[0]=cbr;
    function rhs([cB, cBr]) {
      const dcb = -P_eff * (cB - (cBr / K));
      const dcbr = P_eff * (cB - (cBr / K));
      return [dcb, dcbr];
    }
    for(let i=1;i<steps;i++){
      const s0 = [cb, cbr];
      const k1 = rhs(s0);
      const s1 = [ cb + 0.5*k1[0]*dt, cbr + 0.5*k1[1]*dt ];
      const k2 = rhs(s1);
      const s2 = [ cb + 0.5*k2[0]*dt, cbr + 0.5*k2[1]*dt ];
      const k3 = rhs(s2);
      const s3 = [ cb + k3[0]*dt, cbr + k3[1]*dt ];
      const k4 = rhs(s3);
      cb  = cb  + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
      cbr = cbr + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
      t[i] = i*dt;
      Cb[i] = cb; Cbr[i]=cbr;
    }
    return { t, Cb, Cbr };
  }

  // --- UI refs ---
  const ids = ['MolLogP','MolWt','TPSA','FC','Aromatic','Heavy'];
  const ui = {};
  ids.forEach(id => ui[id] = document.getElementById(id));
  ui.P_eff = document.getElementById('P_eff');
  ui.Tmax  = document.getElementById('Tmax');

  const out = {};
  out.MolLogP = document.getElementById('MolLogP_val');
  out.MolWt = document.getElementById('MolWt_val');
  out.TPSA = document.getElementById('TPSA_val');
  out.FC = document.getElementById('FC_val');
  out.Aromatic = document.getElementById('Aromatic_val');
  out.Heavy = document.getElementById('Heavy_val');

  const drugSel = document.getElementById('drug-select');
  const runBtn = document.getElementById('runBtn');
  const resetBtn = document.getElementById('resetBtn');
  const modelLogbbEl = document.getElementById('model_logbb');
  const expLogbbEl = document.getElementById('exp_logbb');

  // make slider outputs visible and reactive
  ids.forEach(id => {
    if(ui[id]){
      ui[id].addEventListener('input', ()=> out[id].textContent = ui[id].value);
      out[id].textContent = ui[id].value;
    }
  });

  // --- load TSV ---
  let parsedRows = [];
  try{
    const txt = await fetchText('./B3DB_regression.tsv');
    const parsed = parseTSV(txt);
    const hdr = parsed.hdr;
    const rows = parsed.rows;

    // flexible header matching
    function col(candidates){
      for(const c of candidates){
        const exact = hdr.find(h => h === c);
        if(exact) return exact;
      }
      // normalized matching
      for(const c of candidates){
        const norm = c.toLowerCase().replace(/\W/g,'');
        const found = hdr.find(h => h.toLowerCase().replace(/\W/g,'') === norm);
        if(found) return found;
      }
      return null;
    }

    const col_name = col(['compound_name','compound','name']);
    const col_MolLogP = col(['MolLogP','LogP','molLogP']);
    const col_MolWt = col(['MolWt','MolWeight','MolecularWeight']);
    const col_TPSA = col(['TPSA','TopologicalPolarSurfaceArea']);
    const col_FC = col(['FC','FormalCharge','Formal_Charge']);
    const col_Aromatic = col(['Aromatic Rings','Aromatic','aromatic_rings']);
    const col_Heavy = col(['Heavy Atoms','HeavyAtoms','heavy_atoms']);
    const col_logBB = col(['logBB','LogBB','logBB_exp','logBB_experimental']);

    parsedRows = rows.map(r => ({
      name: r[col_name] ?? r[Object.keys(r)[0]] ?? '',
      MolLogP: toNum(r[col_MolLogP]),
      MolWt: toNum(r[col_MolWt]),
      TPSA: toNum(r[col_TPSA]),
      FC: toNum(r[col_FC]),
      Aromatic: toNum(r[col_Aromatic]),
      Heavy: toNum(r[col_Heavy]),
      logBB_exp: col_logBB ? toNum(r[col_logBB]) : NaN
    }));
  }catch(err){
    console.error('Failed to load TSV', err);
    document.getElementById('plot').innerHTML = '<div style="color:#900;padding:16px;">Failed to load B3DB_regression.tsv — check path and CORS.</div>';
    return;
  }

  // populate dropdown
  const names = Array.from(new Set(parsedRows.map(r => r.name))).filter(Boolean).sort();
  drugSel.innerHTML = '';
  names.forEach(n => {
    const o = document.createElement('option'); o.value = n; o.textContent = n; drugSel.appendChild(o);
  });

  // compute OLS but only using rows with experimental logBB defined
  const X = [], y = [];
  parsedRows.forEach(r => {
    if([r.MolLogP,r.MolWt,r.TPSA,r.FC,r.Aromatic,r.Heavy].some(v=> Number.isNaN(v))) return;
    if(Number.isNaN(r.logBB_exp)) return; // train only on rows with experimental logBB
    X.push([1, r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy]);
    y.push(r.logBB_exp);
  });

  let beta = null;
  try{
    if(X.length >= 6){ beta = ols(X,y); console.log('beta', beta); }
    else console.warn('Not enough training rows for OLS: found', X.length);
  }catch(e){
    console.warn('OLS error', e);
    beta = null;
  }

  function predict(features){
    if(!beta) return NaN;
    const vals = [1, features.MolLogP, features.MolWt, features.TPSA, features.FC, features.Aromatic, features.Heavy];
    return vals.reduce((s,v,i)=> s + (beta[i]||0)*v, 0);
  }

  function findRow(name){
    return parsedRows.find(r=> r.name === name);
  }

  function applyRowToSliders(row){
    if(!row) return;
    // guard against NaN: only set sliders when numeric
    if(Number.isFinite(row.MolLogP)) ui.MolLogP.value = row.MolLogP;
    if(Number.isFinite(row.MolWt))   ui.MolWt.value = row.MolWt;
    if(Number.isFinite(row.TPSA))    ui.TPSA.value = row.TPSA;
    if(Number.isFinite(row.FC))      ui.FC.value = row.FC;
    if(Number.isFinite(row.Aromatic))ui.Aromatic.value = row.Aromatic;
    if(Number.isFinite(row.Heavy))   ui.Heavy.value = row.Heavy;
    // update outputs
    ids.forEach(id => out[id].textContent = ui[id].value);
  }

  // update textual logbb display
  function updateLogbbText(){
    const features = {
      MolLogP: +ui.MolLogP.value,
      MolWt: +ui.MolWt.value,
      TPSA: +ui.TPSA.value,
      FC: +ui.FC.value,
      Aromatic: +ui.Aromatic.value,
      Heavy: +ui.Heavy.value
    };
    const model = predict(features);
    const row = findRow(drugSel.value);
    const exp = (row && Number.isFinite(row.logBB_exp)) ? row.logBB_exp : null;
    modelLogbbEl.textContent = Number.isFinite(model) ? model.toFixed(3) : '—';
    expLogbbEl.textContent = exp !== null ? exp.toFixed(3) : '—';
    return {model, exp};
  }

  // simulation + plotting
  function run(){
    const {model, exp} = updateLogbbText();
    const P_eff = toNum(ui.P_eff.value) || 0.15;
    const Tmax = Math.max(1, toNum(ui.Tmax.value) || 24);
    const dt = 0.05;

    const traces = [];
    if(Number.isFinite(model)){
      const s = simulateTwoComp(model, P_eff, Tmax, dt);
      traces.push({ x:s.t, y:s.Cb, name:'Blood (Model)', mode:'lines', line:{width:2, color:'#1f77b4'} });
      traces.push({ x:s.t, y:s.Cbr, name:'Brain (Model)', mode:'lines', line:{width:2, color:'#ff7f0e'} });
    }

    if(Number.isFinite(exp)){
      const s2 = simulateTwoComp(exp, P_eff, Tmax, dt);
      traces.push({ x:s2.t, y:s2.Cb, name:'Blood (Experimental)', mode:'lines', line:{dash:'dashdot', width:2, color:'#1f77b4'} });
      traces.push({ x:s2.t, y:s2.Cbr, name:'Brain (Experimental)', mode:'lines', line:{dash:'dashdot', width:2, color:'#ff7f0e'} });
    }

    // if we have experimental logBB but want to show a marker for observed ratio:
    const row = findRow(drugSel.value);
    if(row && Number.isFinite(row.logBB_exp)){
      // add a marker at t=0 showing brain/blood ratio = 10^logBB
      const ratio = Math.pow(10, row.logBB_exp);
      // normalized representation: if blood=1 then brain=ratio (might exceed 1), so we show a small marker on right-hand axis.
      // Instead, add text annotation showing experimental ratio
      traces.push({ x:[0], y:[ Math.min(1, ratio/(ratio+1)) ], name:'Exp Ratio marker', mode:'markers', marker:{size:8, color:'#2ca02c'} });
    }

    const layout = {
      title: { text: "Blood & Brain concentrations — Model vs Experimental", font:{size:16} },
      xaxis: { title:'Time (h)' },
      yaxis: { title:'Concentration (normalized)', range:[0,1.05] },
      legend: { orientation:'h', y:1.08 },
      margin: { t:70 }
    };

    Plotly.react('plot', traces, layout, {responsive:true});
  }

  // events
  drugSel.addEventListener('change', () => {
    const r = findRow(drugSel.value);
    if(r) applyRowToSliders(r);
    updateLogbbText();
    // optionally auto-run
    run();
  });

  ids.forEach(id => ui[id].addEventListener('input', ()=>{
    out[id].textContent = ui[id].value;
    updateLogbbText();
  }));

  runBtn.addEventListener('click', run);
  resetBtn.addEventListener('click', ()=>{
    const r = findRow(drugSel.value);
    if(r) applyRowToSliders(r);
    updateLogbbText();
  });

  // initialize selection
  if(names.length>0){
    drugSel.value = names[0];
    const r = findRow(names[0]);
    if(r) applyRowToSliders(r);
  }

  // initial run
  run();

})(); // end IIFE

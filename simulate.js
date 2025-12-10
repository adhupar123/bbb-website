/* simulate.js
 - Fetches ./B3DB_regression.tsv
 - Computes OLS regression (logBB ~ MolLogP + MolWt + TPSA + FC + Aromatic + Heavy)
 - Hooks up UI: dropdown and sliders
 - Runs a two-compartment Fick model with simple numerical integration
 - Plots Model vs Experimental results using Plotly
*/

(async function(){
  // helper: parse TSV
  async function loadTSV(path){
    const res = await fetch(path);
    if(!res.ok) throw new Error('Could not fetch TSV at ' + path);
    const txt = await res.text();
    const lines = txt.trim().split(/\r?\n/);
    const hdr = lines[0].split('\t');
    const data = lines.slice(1).map(l => {
      const vals = l.split('\t');
      const obj = {};
      hdr.forEach((h,i)=> obj[h] = vals[i]);
      return obj;
    });
    return {hdr, data};
  }

  // Numeric helpers
  function toNum(v){ return v === undefined || v === '' ? NaN : +v; }

  function transpose(A){
    return A[0].map((_,i)=> A.map(row=> row[i]));
  }
  // multiply matrix A (m x n) by B (n x p)
  function matMul(A,B){
    const m=A.length, n=A[0].length, p=B[0].length;
    const C = Array.from({length:m}, ()=> Array(p).fill(0));
    for(let i=0;i<m;i++){
      for(let k=0;k<n;k++){
        const aik = A[i][k];
        for(let j=0;j<p;j++) C[i][j] += aik * B[k][j];
      }
    }
    return C;
  }
  // invert symmetric matrix via numeric JS (use Gauss-Jordan)
  function invertMatrix(M){
    const n = M.length;
    // clone
    const A = M.map(r=> r.slice());
    const I = Array.from({length:n}, (_,i)=> Array.from({length:n}, (__,j)=> i===j?1:0));
    for(let i=0;i<n;i++){
      // pivot
      let pivot = A[i][i];
      if(Math.abs(pivot) < 1e-12){
        // find row to swap
        let swap = i+1;
        while(swap < n && Math.abs(A[swap][i]) < 1e-12) swap++;
        if(swap === n) throw new Error("Singular matrix");
        [A[i], A[swap]] = [A[swap], A[i]];
        [I[i], I[swap]] = [I[swap], I[i]];
        pivot = A[i][i];
      }
      // normalize row
      for(let j=0;j<n;j++){ A[i][j] /= pivot; I[i][j] /= pivot; }
      // eliminate other rows
      for(let r=0;r<n;r++){
        if(r===i) continue;
        const f = A[r][i];
        if(f===0) continue;
        for(let c=0;c<n;c++){ A[r][c] -= f * A[i][c]; I[r][c] -= f * I[i][c]; }
      }
    }
    return I;
  }

  // OLS: compute beta = (X^T X)^{-1} X^T y
  function ols(X, y){
    // X: n x p, y: n x 1
    const Xt = transpose(X);
    const XtX = matMul(Xt, X); // p x p
    const XtXinv = invertMatrix(XtX); // p x p
    const Xty = matMul(Xt, y.map(v=>[v])); // p x 1
    const betaMat = matMul(XtXinv, Xty); // p x 1
    return betaMat.map(r => r[0]); // p coefficients
  }

  // integrate two-compartment exchange (simple explicit RK4)
  function simulateTwoComp(logBB, P_eff=0.15, Tmax=24, dt=0.1){
    // K = partition from brain to blood: K = 10^(logBB)
    const K = Math.pow(10, logBB);
    const steps = Math.max(2, Math.floor(Tmax/dt)+1);
    const t = new Array(steps);
    const Cb = new Array(steps);
    const Cbr = new Array(steps);
    // initial concentrations (normalized)
    let cb = 1.0, cbr = 0.0;
    t[0]=0; Cb[0]=cb; Cbr[0]=cbr;
    for(let i=1;i<steps;i++){
      const time = i*dt;
      // RHS
      function rhs(state){
        const [cB, cBr] = state;
        const dcb = -P_eff * (cB - cBr / K);
        const dcbr =  P_eff * (cB - cBr / K);
        return [dcb, dcbr];
      }
      // RK4
      const s0 = [cb, cbr];
      const k1 = rhs(s0);
      const s1 = [ cb + 0.5*dt*k1[0], cbr + 0.5*dt*k1[1] ];
      const k2 = rhs(s1);
      const s2 = [ cb + 0.5*dt*k2[0], cbr + 0.5*dt*k2[1] ];
      const k3 = rhs(s2);
      const s3 = [ cb + dt*k3[0], cbr + dt*k3[1] ];
      const k4 = rhs(s3);
      cb = cb + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
      cbr = cbr + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
      t[i]=time; Cb[i]=cb; Cbr[i]=cbr;
    }
    return {t, Cb, Cbr};
  }

  // UI elements
  const ids = ['MolLogP','MolWt','TPSA','FC','Aromatic','Heavy'];
  const ui = {};
  ids.forEach(id => ui[id] = document.getElementById(id));
  ui['P_eff'] = document.getElementById('P_eff');
  ui['Tmax'] = document.getElementById('Tmax');
  const outEls = {
    MolLogP: document.getElementById('MolLogP_val'),
    MolWt: document.getElementById('MolWt_val'),
    TPSA: document.getElementById('TPSA_val'),
    FC: document.getElementById('FC_val'),
    Aromatic: document.getElementById('Aromatic_val'),
    Heavy: document.getElementById('Heavy_val')
  };
  const drugSel = document.getElementById('drug-select');
  const runBtn = document.getElementById('runBtn');
  const resetBtn = document.getElementById('resetBtn');
  const modelLogbbEl = document.getElementById('model_logbb');
  const expLogbbEl = document.getElementById('exp_logbb');

  // attach slider output updates
  ids.forEach(k=>{
    ui[k].addEventListener('input', ()=> {
      outEls[k].textContent = ui[k].value;
    });
    // set outputs initially
    outEls[k].textContent = ui[k].value;
  });

  // load TSV
  let tsv;
  try{
    tsv = await loadTSV('./B3DB_regression.tsv');
  }catch(e){
    console.error(e);
    document.getElementById('plot').innerHTML = '<div style="color:#a00">Error: could not load B3DB_regression.tsv. Make sure the file is reachable at ./B3DB_regression.tsv</div>';
    return;
  }
  const rows = tsv.data;
  // expected column names - try flexible matching
  // We'll look for: compound_name, MolLogP, MolWt, TPSA, FC, Aromatic Rings, Heavy Atoms, logBB
  function findCol(names){
    for(const n of names){
      if(tsv.hdr.includes(n)) return n;
      // try trimmed versions
      const found = tsv.hdr.find(h => h.toLowerCase().replace(/\s+/g,'') === n.toLowerCase().replace(/\s+/g,''));
      if(found) return found;
    }
    return null;
  }
  const col_compound = findCol(['compound_name','compound','name']);
  const col_MolLogP = findCol(['MolLogP','MolLogP_logP','molLogP','LogP']);
  const col_MolWt = findCol(['MolWt','MolecularWeight','MolWeight']);
  const col_TPSA = findCol(['TPSA','TopologicalPolarSurfaceArea','tpsa']);
  const col_FC = findCol(['FC','FormalCharge','Formal_Charge']);
  const col_Aromatic = findCol(['Aromatic Rings','Aromatic_Rings','Aromatic']);
  const col_Heavy = findCol(['Heavy Atoms','Heavy_Atoms','HeavyAtoms']);
  const col_logBB = findCol(['logBB','LogBB','logBB_exp','logBB_experimental','logBB_exp']);

  // Build usable data array and populate dropdown
  const drugs = [];
  const numericRows = [];
  for(const r of rows){
    const obj = {
      compound: r[col_compound] ?? '',
      MolLogP: toNum(r[col_MolLogP]),
      MolWt: toNum(r[col_MolWt]),
      TPSA: toNum(r[col_TPSA]),
      FC: toNum(r[col_FC]),
      Aromatic: toNum(r[col_Aromatic]),
      Heavy: toNum(r[col_Heavy]),
      logBB_exp: col_logBB ? toNum(r[col_logBB]) : NaN
    };
    numericRows.push(obj);
    drugs.push(obj.compound);
  }

  // fill dropdown (unique)
  const uniqueNames = Array.from(new Set(drugs)).filter(Boolean).sort();
  uniqueNames.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n;
    opt.textContent = n;
    drugSel.appendChild(opt);
  });

  // Build X and y for regression (skip rows with missing values)
  const X = [];
  const y = [];
  for(const r of numericRows){
    if([r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy].some(v=> Number.isNaN(v))) continue;
    X.push([1, r.MolLogP, r.MolWt, r.TPSA, r.FC, r.Aromatic, r.Heavy]); // intercept + features
    y.push(isFinite(r.logBB_exp) ? r.logBB_exp : NaN);
  }
  // filter rows where y is NaN? In the notebook they likely trained on rows where logBB exists.
  const X_fit = [], y_fit = [];
  for(let i=0;i<X.length;i++){ if(!Number.isNaN(y[i])){ X_fit.push(X[i]); y_fit.push(y[i]); } }
  let beta;
  try{
    beta = ols(X_fit, y_fit); // coefficients: intercept, MolLogP, MolWt, TPSA, FC, Aromatic, Heavy
    console.log('beta', beta);
  }catch(e){
    console.warn('OLS failed', e);
    beta = null;
  }

  // compute predicted logBB from feature vector
  function predictLogBB(features){ // features: {MolLogP, MolWt, TPSA, FC, Aromatic, Heavy}
    if(!beta) return NaN;
    const vals = [1, features.MolLogP, features.MolWt, features.TPSA, features.FC, features.Aromatic, features.Heavy];
    let s = 0;
    for(let i=0;i<beta.length;i++) s += beta[i]*vals[i];
    return s;
  }

  // fill sliders when drug selected
  function setSlidersFromRow(row){
    if(!row) return;
    ui.MolLogP.value = row.MolLogP ?? ui.MolLogP.value;
    ui.MolWt.value = row.MolWt ?? ui.MolWt.value;
    ui.TPSA.value = row.TPSA ?? ui.TPSA.value;
    ui.FC.value = row.FC ?? ui.FC.value;
    ui.Aromatic.value = row.Aromatic ?? ui.Aromatic.value;
    ui.Heavy.value = row.Heavy ?? ui.Heavy.value;
    // update visible outputs
    ids.forEach(k => outEls[k].textContent = ui[k].value);
  }

  // get numeric row by compound name
  function getRowByName(name){
    return numericRows.find(r => r.compound === name);
  }

  // event: when user picks drug
  drugSel.addEventListener('change', ()=>{
    const name = drugSel.value;
    const row = getRowByName(name);
    setSlidersFromRow(row);
    // update model/experimental displays
    updateModelText();
  });

  // update model/exp text
  function updateModelText(){
    const features = {
      MolLogP: toNum(ui.MolLogP.value),
      MolWt: toNum(ui.MolWt.value),
      TPSA: toNum(ui.TPSA.value),
      FC: toNum(ui.FC.value),
      Aromatic: toNum(ui.Aromatic.value),
      Heavy: toNum(ui.Heavy.value)
    };
    const model_logbb = predictLogBB(features);
    const selected = getRowByName(drugSel.value);
    const exp_logbb = selected && isFinite(selected.logBB_exp) ? selected.logBB_exp : null;
    modelLogbbEl.textContent = Number.isFinite(model_logbb) ? model_logbb.toFixed(2) : '—';
    expLogbbEl.textContent = exp_logbb !== null ? exp_logbb.toFixed(2) : '—';
    return {model_logbb, exp_logbb};
  }

  // Run simulation and plot
  function runAndPlot(){
    const {model_logbb, exp_logbb} = updateModelText();
    const P_eff_val = toNum(ui.P_eff.value) || 0.15;
    const Tmax = Math.max(1, toNum(ui.Tmax.value) || 24);
    const dt = 0.05;

    const traces = [];

    if(Number.isFinite(model_logbb)){
      const out = simulateTwoComp(model_logbb, P_eff_val, Tmax, dt);
      traces.push({
        x: out.t, y: out.Cb, name:'Blood (Model)', mode:'lines', line:{dash:'solid', width:2}
      });
      traces.push({
        x: out.t, y: out.Cbr, name:'Brain (Model)', mode:'lines', line:{dash:'solid', width:2}
      });
    }

    if(Number.isFinite(exp_logbb)){
      const out2 = simulateTwoComp(exp_logbb, P_eff_val, Tmax, dt);
      traces.push({
        x: out2.t, y: out2.Cb, name:'Blood (Experimental)', mode:'lines', line:{dash:'dash', width:2}
      });
      traces.push({
        x: out2.t, y: out2.Cbr, name:'Brain (Experimental)', mode:'lines', line:{dash:'dash', width:2}
      });
    }

    const layout = {
      title: `Fick's Law Simulation — Model LogBB=${Number.isFinite(model_logbb)?model_logbb.toFixed(2):'—'}  Exp LogBB=${Number.isFinite(exp_logbb)?exp_logbb.toFixed(2):'—'}`,
      xaxis: {title:'Time (h)'},
      yaxis: {title:'Concentration (normalized)', range:[0,1.05]},
      legend: {orientation:'h'},
      margin:{t:60}
    };

    Plotly.newPlot('plot', traces, layout, {responsive:true});
  }

  // attach run and reset
  runBtn.addEventListener('click', runAndPlot);
  resetBtn.addEventListener('click', ()=>{
    const row = getRowByName(drugSel.value);
    if(row) setSlidersFromRow(row);
    updateModelText();
  });

  // initialize: choose first drug (if any)
  if(uniqueNames.length>0){
    drugSel.value = uniqueNames[0];
    const row = getRowByName(uniqueNames[0]);
    setSlidersFromRow(row);
    updateModelText();
  } else {
    // no drugs - leave UI as-is
  }

  // initial plot
  runAndPlot();

})();

// Rebuild: npm install --prefix /tmp/opd-animation-tools @resvg/resvg-js gifenc
// NODE_PATH=/tmp/opd-animation-tools/node_modules node scripts/render-overview-animations.cjs
const fs = require('node:fs');
const path = require('node:path');
const { Resvg } = require('@resvg/resvg-js');
const { GIFEncoder, quantize, applyPalette } = require('gifenc');
const out = path.join(__dirname, '../docs/assets/animations');
const sheet = fs.readFileSync(path.join(out, 'characters.png')).toString('base64');
const W=800, H=500, FPS=12, SECONDS=12;
const wrap = content => `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#eaf2f5"/><stop offset="1" stop-color="#faf6e9"/></linearGradient><linearGradient id="sand" x2="0" y2="1"><stop stop-color="#fcf0d5"/><stop offset="1" stop-color="#efd8aa"/></linearGradient></defs>${content}</svg>`;
const render = svg => new Resvg(svg, {font:{defaultFontFamily:'Arial',loadSystemFonts:false,fontFiles:['/System/Library/Fonts/Supplemental/Arial.ttf']}}).render();
// Extract each isolated character from the generated transparent sheet without redrawing it.
for (const [name,box] of [['teacher',[178,15,612,978]],['student',[885,373,445,620]]]) {
  const [x,y,w,h]=box;
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${w}" height="${h}" viewBox="${x} ${y} ${w} ${h}"><image width="1536" height="1024" xlink:href="data:image/png;base64,${sheet}"/></svg>`;
  fs.writeFileSync(path.join(out,`${name}.png`),render(svg).asPng());
}
const sprites = Object.fromEntries(['teacher','student'].map(n=>[n,fs.readFileSync(path.join(out,n+'.png')).toString('base64')]));
const shelter = fs.readFileSync(path.join(out,'shelter.png')).toString('base64');
const text=(x,y,s,size=16,color='#4e4c43',extra='')=>`<text x="${x}" y="${y}" font-family="Arial, sans-serif" font-size="${size}" fill="${color}" ${extra}>${s}</text>`;
const line=(d,color,width,extra='')=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
function person(name,x,y,height,flip=false,bob=0){
 const width=height*(name==='teacher'?612/978:445/620);
 return `<g transform="translate(${x},${y+bob})"><ellipse cx="0" cy="0" rx="${width*.31}" ry="5" fill="#665b46" opacity=".12"/><g transform="scale(${flip?-1:1},1)"><image x="${-width/2}" y="${-height}" width="${width}" height="${height}" xlink:href="data:image/png;base64,${sprites[name]}"/></g></g>`;
}
function diamond(x,y,scale=1){return `<g transform="translate(${x},${y}) scale(${scale})" stroke="#327799" stroke-width="1.5"><path d="M-20 -9L-10 -21H10L20 -9L0 16Z" fill="#88d8f3"/><path d="M-20 -9H20M-10 -21L-7 -9L0 16L7 -9L10 -21" fill="none"/><path d="M-10 -21L0 -9L10 -21M-7 -9H7" fill="none" stroke="#e8fbff"/></g>`;}
function tree(x,y,s=1){return `<g transform="translate(${x},${y}) scale(${s})"><path d="M0 0V-38" stroke="#8d7c57" stroke-width="5"/><path d="M0 -17L-12 -30M0 -24L13 -37" stroke="#8d7c57" stroke-width="3"/><path d="M-20 -33Q-30 -48-13 -55Q-14 -77 4 -73Q24 -76 23 -55Q40 -42 20 -29Q8 -22-1 -29Q-13 -22-20 -33" fill="#bdcb9b" stroke="#8a9f72" stroke-width="2"/></g>`;}
function building(x,y,w,h,c){let windows='';for(let row=0;row<Math.floor((h-25)/25);row++)for(let col=0;col<Math.floor((w-12)/22);col++)windows+=`<rect x="${x+12+col*22}" y="${y-h+14+row*25}" width="9" height="13" fill="#8fa6ad" stroke="#647e86" stroke-width="1"/>`;return `<path d="M${x} ${y-h}L${x+w} ${y-h}L${x+w+13} ${y-h-10}L${x+13} ${y-h-10}Z" fill="#ede4cd" stroke="#b6ab94"/><path d="M${x+w} ${y-h}L${x+w+13} ${y-h-10}V${y-7}L${x+w} ${y}Z" fill="#b7b9a8" stroke="#a4a38e"/><rect x="${x}" y="${y-h}" width="${w}" height="${h}" fill="${c}" stroke="#a5a592" stroke-width="1.5"/>${windows}<rect x="${x+w/2-7}" y="${y-23}" width="14" height="23" fill="#8c8b7d"/>`;}
const cityRoutes=['M220 385 C310 340 315 260 420 255 S600 290 680 225','M220 385 C300 400 345 424 460 410 S610 385 720 410','M220 385 C260 300 230 210 315 160 S460 145 500 100'];
function cityBackground(){return `<rect width="800" height="500" fill="#fbfaf5"/><path d="M0 0H800V180Q400 150 0 190Z" fill="url(#sky)"/>${building(20,165,70,100,'#e3d4b8')}${building(105,150,60,120,'#c5d3d5')}${building(187,171,75,93,'#e5dfc9')}${building(355,150,66,110,'#d6dcbf')}${building(525,182,63,132,'#c6d5dc')}${building(608,164,73,100,'#e6d8b4')}${building(711,180,67,143,'#d5d9cb')}<path d="M0 209Q140 185 260 210T530 200T800 211V500H0Z" fill="#f4efdf"/>${cityRoutes.map(d=>line(d,'#d5ceb9',39)+line(d,'#fffcf1',34)).join('')}${cityRoutes.map(d=>line(d,'#cbc8ba',2,'stroke-dasharray="5 9"')).join('')}${tree(69,240,.9)}${tree(445,205,.8)}${tree(758,315,.8)}${tree(337,451,.8)}<path d="M455 331H532" stroke="#a69a7d" stroke-width="8"/><path d="M465 331V347M522 331V347" stroke="#a69a7d" stroke-width="4"/><ellipse cx="680" cy="225" rx="32" ry="11" fill="#d7dfce"/><path d="M655 213H705V226Q680 240 655 226Z" fill="#d0d5c9" stroke="#a6b4ac"/>${text(32,35,'CITY',13,'#6c7c78','letter-spacing="2"')}`;}
function cactus(x,y,s=1){return `<g transform="translate(${x},${y}) scale(${s})"><path d="M0 0V-54M0 -24H-16V-41M0 -16H17V-37" fill="none" stroke="#889265" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"/><path d="M0 -4V-49" stroke="#b4bb89" stroke-width="2"/></g>`;}
// One shared cubic path drives both the drawing and the student's motion.
const desertEntry=[[220,385],[275,390],[320,397],[350,370]];
const desertLoop=[
 [[350,370],[315,333],[358,266],[420,270]],
 [[420,270],[475,267],[526,226],[586,263]],
 [[586,263],[680,301],[685,368],[622,385]],
 [[622,385],[568,401],[513,365],[462,344]],
 [[462,344],[399,310],[372,357],[408,407]],
 [[408,407],[447,460],[549,423],[571,372]],
 [[571,372],[608,287],[692,280],[708,343]],
 [[708,343],[730,420],[649,448],[579,421]],
 [[579,421],[515,398],[524,321],[459,302]],
 [[459,302],[394,281],[429,414],[350,370]],
];
const cubicPath = segments => `M${segments[0][0].join(' ')} `+segments.map(p=>`C${p.slice(1).map(v=>v.join(' ')).join(' ')}`).join(' ');
const desertRoute=cubicPath([desertEntry,...desertLoop]);
const desertCandidates=[
 'M220 385 C246 297 315 239 410 215 S606 195 689 175',
 'M220 385 C242 446 326 449 392 449 S546 451 613 454 S711 437 755 403',
];
function desertBackground(){return `<rect width="800" height="500" fill="url(#sand)"/><circle cx="695" cy="60" r="29" fill="#f1dca0"/><path d="M0 151L54 141L72 97L118 97L143 142L206 152L235 132L259 132L290 160L365 166L386 119L424 119L447 161L537 152L575 135L600 135L625 171L800 152V223H0Z" fill="#e8c894"/><path d="M0 197Q145 154 308 198T615 192T800 196V500H0Z" fill="#f4e2be"/><path d="M0 245Q125 204 269 231M568 204Q665 182 800 223M39 455Q150 433 251 453M597 464Q705 442 800 463" fill="none" stroke="#dfc697" stroke-width="2"/>${desertCandidates.map(d=>line(d,'#e2c99d',24)+line(d,'#f9edce',19)+line(d,'#bda782',2,'stroke-dasharray="5 8"')).join('')}${line(desertRoute,'#e8c99c',23)}${line(desertRoute,'#c5a575',2,'stroke-dasharray="5 9"')}<path d="M447 342L455 313L472 293L499 300L520 335L535 352Q487 366 438 351Z" fill="#c4af85" stroke="#aa946f" stroke-width="2"/><path d="M472 293L482 323L465 350M482 323L516 336" fill="none" stroke="#ac966f" stroke-width="2"/>${cactus(745,318,.9)}${cactus(278,260,.7)}${cactus(97,420,.65)}<g fill="#c3a77d"><ellipse cx="582" cy="456" rx="8" ry="4"/><ellipse cx="311" cy="452" rx="7" ry="3"/><ellipse cx="691" cy="284" rx="8" ry="4"/></g>${text(32,35,'DESERT',13,'#96774b','letter-spacing="2"')}${text(365,203,'Route A',12,'#978260')}${text(329,439,'Route C',12,'#978260')}${text(689,94,'Shelter',14,'#8c805e','text-anchor="middle"')}<svg x="619" y="111" width="140" height="61" viewBox="296 190 1186 519"><image width="1774" height="887" xlink:href="data:image/png;base64,${shelter}"/></svg>`;}
const cityBg=cityBackground(), desertBg=desertBackground();
const bezier=(p0,p1,p2,p3,t)=>p0.map((_,i)=>(1-t)**3*p0[i]+3*(1-t)**2*t*p1[i]+3*(1-t)*t*t*p2[i]+t**3*p3[i]);
function cityPoint(u){return u<.5?bezier([220,385],[310,340],[315,260],[420,255],u*2):bezier([420,255],[525,250],[600,290],[680,225],u*2-1);}
let desertSamples;
function desertPoint(u){
 if(u<.16)return bezier(...desertEntry,u/.16);
 if(!desertSamples){
  desertSamples=[{p:desertLoop[0][0],distance:0}];
  for(const segment of desertLoop)for(let i=1;i<=100;i++){
   const p=bezier(...segment,i/100),last=desertSamples.at(-1);
   desertSamples.push({p,distance:last.distance+Math.hypot(p[0]-last.p[0],p[1]-last.p[1])});
  }
 }
 const fraction=((u-.16)/1.0)%1;
 const target=fraction*desertSamples.at(-1).distance;
 const index=desertSamples.findIndex(v=>v.distance>=target);
 if(index<=0)return desertSamples[0].p;
 const a=desertSamples[index-1],b=desertSamples[index];
 const r=(target-a.distance)/(b.distance-a.distance);
 return a.p.map((v,i)=>v+(b.p[i]-v)*r);
}
function scene(kind,t){
 const city=kind==='city', walking=t>=3&&(!city||t<9), selected=t>=1.4;
 const u=Math.max(0,Math.min(1,(t-3)/6));
 const [x,y]=city?cityPoint(u):desertPoint(Math.max(0,(t-3)/6));
 const before=city?cityPoint(Math.max(0,u-.003)):desertPoint(Math.max(0,(t-3)/6-.003));
 const flip=walking&&before[0]>x;
 const arrived=city&&t>=9;
 const color=city?'#4c8875':'#b47744';
 const phase=t<1.4?'Several paths are available.':t<3?'The teacher chooses a route.':t<9?(city?'The student follows the chosen route.':'The student follows the same route...'):(city?'The student reaches the diamond.':'...and keeps going in circles.');
 let route=selected?line(city?cityRoutes[0]:desertRoute,color,5,'stroke-dasharray="10 8"'):'';
 let badge=selected?`<rect x="180" y="193" width="139" height="29" rx="5" fill="white" stroke="${color}"/>${text(249,213,'Take this route →',13,color,'text-anchor="middle"')}`:'';
 const stars=arrived?Array.from({length:5},(_,i)=>{let a=i*Math.PI*2/5+t*.5;let sx=x+Math.cos(a)*51,sy=y-61+Math.sin(a)*47;return `<path d="M${sx-4} ${sy}H${sx+4}M${sx} ${sy-4}V${sy+4}" stroke="#d9aa46" stroke-width="2"/>`;}).join(''):'';
 let student=person('student',x,y,91,flip,walking?-Math.abs(Math.sin(t*13))*3:0);
 return wrap((city?cityBg:desertBg)+route+person('teacher',128,370,178)+text(120,393,'Teacher',14,'#655e48','text-anchor="middle"')+badge+(!city?'':!arrived?diamond(680,195):'')+student+(arrived?diamond(x+28,y-43,.65)+stars:'')+(t<3?text(220,412,'Student',14,'#426c83','text-anchor="middle"'):'')+`<rect y="465" width="800" height="35" fill="${city?'#f5f7f2':'#fcf4e4'}"/>`+text(400,488,phase,16,'#494b43','text-anchor="middle"'));
}
if (require.main === module) for(const kind of (process.argv[2] ? [process.argv[2]] : ['city','desert'])){
 const gif=GIFEncoder();
 const previewTimes=[0,2,6,10];
 for(const t of previewTimes)fs.writeFileSync(path.join(out,`${kind}-preview-${t}.png`),render(scene(kind,t)).asPng());
 const samples=previewTimes.map(t=>render(scene(kind,t)).pixels);
 const combined=new Uint8Array(samples.reduce((n,a)=>n+a.length,0));let offset=0;for(const a of samples){combined.set(a,offset);offset+=a.length;}
 const palette=quantize(combined,256);
 for(let i=0;i<FPS*SECONDS;i++){
  if(i%36===0)console.log(kind, 'frame', i);
  const frame=render(scene(kind,i/FPS));
  gif.writeFrame(applyPalette(frame.pixels,palette),W,H,{palette:i===0?palette:undefined,delay:i%3===0?90:80,repeat:0});
 }
 gif.finish();fs.writeFileSync(path.join(out,`${kind}.gif`),gif.bytes());
 fs.copyFileSync(path.join(out,`${kind}-preview-10.png`),path.join(out,`${kind}-poster.png`));
 console.log(`${kind}: ${FPS*SECONDS} frames, ${gif.bytes().length} bytes`);
}

module.exports = { scene, render };

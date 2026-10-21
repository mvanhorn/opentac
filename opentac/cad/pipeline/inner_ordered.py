"""Reserve the whole ordered channel before emitting any copper.

Order sources bottom-up, with left-to-right ordering for pieces at the same
height. A backward interval pass first refuses sources with no downward
channel to the connector. Give the remaining order the remaining connector
pads from left to right; refused nets retain their original pads. Every wire
has a lane variable at each via row below its source and at changes in the
connector channel. Domains certify via clearance, the source stub and a
rectangular corridor between levels. Adjacent wires reserve disjoint spans.

A gap with K horizontal tracks divides the global wire order into blocks of
K. Cross-height inequalities keep neighbouring blocks' vertical legs apart;
each block needs at most K tracks, regardless of how far its wires move.
Blocks are never renumbered as rows enter. With no tracks, x cannot change.
All constraints are monotone lower bounds: start at each domain's first slot
and propagate increases to a fixed point. Each slot only moves right, so
exhaustion is a finite refusal, without path search or a cost function.
Draw the certified jogs in precedence order. Below the vias, channel levels
include polygon vertices and lanes include pad positions; this reserves the
connector approach as part of the same problem.

For bridges, group overlapping source-piece intervals, require common via
columns and complete side strips, and nest adjacent-group U paths by height.
The caller checks all complete, quantized paths, stubs, vias and the allowed
region independently before accepting a bundle. A failed certificate means
this construction failed, not that the board is physically unroutable; the
sweep (inner_engine.py) remains a checked fallback. Selected with
OT_INNER_WIRE=ordered (route_inner.py).
"""
from bisect import bisect_left
from collections import deque
import math
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from .inner_engine import Lattice, intervals as raw_intervals

GUARD = .002


def ordered_tracks(jogs, heights, spacing):
    out={}
    for sign in [-1,1]:
        placed=[]
        for k,x0,x1 in sorted((j for j in jogs if (j[2]-j[1])*sign>0),key=lambda j:sign*j[1]):
            lo,hi=sorted([x0,x1]);rank=0
            for u,v,t in placed:
                if u<hi+spacing and lo<v+spacing:rank=max(rank,t+1)
            if rank>=len(heights):return None
            out[k]=heights[rank];placed.append((lo,hi,rank))
    return out


def allowed_region(field, board, ypad):
    """Field plus only the region below its lower vertical silhouette."""
    roof=field.bounds[3]+1
    xlo, _, xhi, _ = field.bounds
    def edge_bottom(x):
        return field.intersection(LineString([(x,-1e4),(x,roof)])).bounds[1]
    shadows=[box(-1e4,edge_bottom(xlo),xlo,roof),
             box(xhi,edge_bottom(xhi),1e4,roof)]
    for p in getattr(field,'geoms',[field]):
        xy=list(p.exterior.coords)
        for (x,y),(u,v) in zip(xy,xy[1:]):
            if abs(x-u)>1e-9:
                shadows.append(Polygon([(x,y),(u,v),(u,roof),(x,roof)]))
    shadow=unary_union(shadows)
    return field.union(board.difference(shadow)).intersection(box(-1e4,ypad,1e4,roof)).buffer(0)


def intervals(F, y):
    merged=[]
    for lo,hi in raw_intervals(F,y):
        if merged and lo<=merged[-1][1]+1e-9:
            merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    return merged


def common_intervals(F, lo, hi, vertices):
    """Project the forbidden area of the gap onto x; take its complement.

    A connected polygon's x projection is its bounding interval. Subtracting
    all such intervals certifies entire rectangles, including interior holes
    whose horizontal edges would be missed by cuts at vertex heights alone.
    """
    left, _, right, _ = F.bounds
    outside = box(left, lo, right, hi).difference(F)
    blocked = sorted((g.bounds[0], g.bounds[2])
                     for g in getattr(outside, 'geoms', [outside]) if not g.is_empty)
    result = []
    for start, end in blocked:
        if start > left + 2*GUARD:
            result.append((left, start))
        left = max(left, end)
    if right > left + 2*GUARD:
        result.append((left, right))
    return result


def solve(rows, X, Yr, F, a, lane, pitch):
    a+=GUARD;lane+=GUARD
    pads=sorted(((r['pad_num'],r['pad_xy']) for r in rows),key=lambda q:q[1][0])
    rows=sorted(rows,key=lambda r:r['order'])
    lat=Lattice(X,Yr,a,lane,pitch=pitch)
    ypad=max(float(r['pad_xy'][1]) for r in rows)+lane
    via_levels=set(round(float(y),6) for y in Yr)
    vertices=sorted({q[1] for p in getattr(F,'geoms',[F])
                     for ring in [p.exterior, *p.interiors] for q in ring.coords})
    ys=sorted(via_levels | {float(y) for y in vertices if ypad<y<min(via_levels)},reverse=True)+[ypad]
    def gap_tracks(k):
        lo,hi=ys[k],ys[k-1]
        lower=a if lo in via_levels else lane/2
        upper=a if hi in via_levels else lane/2
        return np.arange(lo+lower,hi-upper+1e-9,lane).tolist()
    corridors=[None]+[common_intervals(F,ys[k],ys[k-1],vertices) for k in range(1,len(ys))]
    tracks=[[]]+[gap_tracks(k) for k in range(1,len(ys))]
    reach=[0]+[(math.inf if t else 0) for t in tracks[1:]]
    # Domain arrays: x, source-stub left, source-stub right, tap.
    ds=[];meta=[];at={};by_row=[[] for y in ys]
    def mask(xs, cuts):
        ok=np.zeros(len(xs),dtype=bool)
        for lo,hi in cuts:ok |= (xs>=lo+GUARD)&(xs<=hi-GUARD)
        return ok
    base=[]
    for k,y in enumerate(ys):
        xs=lat.lanes if y>=min(via_levels) else np.unique(np.r_[np.arange(F.bounds[0]+GUARD,F.bounds[2],lane),[p[1][0] for p in pads]])
        keep=mask(xs,intervals(F,y))
        if k<len(ys)-1:keep &= mask(xs,corridors[k+1])
        base.append(xs[keep])
    # Single-wire reachability is a backward interval pass, before any
    # pad reassignment or multi-wire inequalities. A source in a disconnected
    # lower arm cannot travel downward to the connector; refuse only that net.
    connected=[None]*len(ys)
    connected[-1]=np.array([p[1][0] for p in pads])
    for k in range(len(ys)-2,-1,-1):
        below=connected[k+1]
        cuts=[(lo,hi) for lo,hi in corridors[k+1]
              if np.any((below>=lo+GUARD)&(below<=hi-GUARD))]
        xs=base[k][mask(base[k],cuts)]
        if not tracks[k+1]:
            xs=xs[np.isin(np.round(xs,6),np.round(below,6))]
        connected[k]=xs
    refused={}
    for r in rows:
        birth=min(range(len(ys)-1),key=lambda k:abs(ys[k]-r['y']))
        xs=connected[birth]
        vv=np.asarray(r['vias'])
        if not len(xs) or not np.any(abs(xs[:,None]-vv[None,:])<=pitch-a+1e-6):
            refused[r['net']]='no downward channel from its vias to the connector'
    rows=[r for r in rows if r['net'] not in refused]
    if not rows:
        return {},dict(reason='no reachable source',refused=refused)
    pads=sorted(((r['pad_num'],r['pad_xy']) for r in rows),key=lambda q:q[1][0])
    rows=[dict(r,pad_num=p[0],pad_xy=p[1]) for r,p in zip(rows,pads)]
    for ni,r in enumerate(rows):
        birth=min(range(len(ys)-1),key=lambda k:abs(ys[k]-r['y']))
        for k in range(birth,len(ys)):
            xs=connected[k] if k<len(ys)-1 else np.array([float(r['pad_xy'][0])])
            if k>birth:xs=xs[mask(xs,corridors[k])]
            tap=xs.copy()
            if k==birth and len(xs):
                vv=np.sort(r['vias']);j=np.searchsorted(vv,xs)
                left=vv[np.maximum(j-1,0)];right=vv[np.minimum(j,len(vv)-1)]
                tap=np.where(abs(left-xs)<=abs(right-xs),left,right)
                segments=np.stack([np.column_stack([tap,np.full(len(xs),r['y'])]),
                                   np.column_stack([xs,np.full(len(xs),ys[k])])],axis=1)
                keep=(abs(tap-xs)<=pitch-a+1e-6)&shapely.covers(F,shapely.linestrings(segments))
                xs,tap=xs[keep],tap[keep]
            cs=np.column_stack([xs,np.minimum(tap,xs),np.maximum(tap,xs),tap])
            if not len(cs):return {},dict(reason='empty domain',net=r['net'],row=k)
            ident=len(ds);ds.append(np.asarray(cs));meta.append((ni,k));at[ni,k]=ident;by_row[k].append(ident)
    outgoing=[[] for d in ds]
    for ids in by_row:
        for i,j in zip(ids,ids[1:]):outgoing[i].append((j,'order',None,None))
    for (ni,k),i in at.items():
        if (ni,k+1) in at:
            j=at[ni,k+1]
            outgoing[i].append((j,'reach',reach[k+1],corridors[k+1]))
            outgoing[j].append((i,'reach',reach[k+1],corridors[k+1]))
    # A gap carries K ordered moves at once. Reserve disjoint blocks of K
    # in the global order; cross-height bounds keep one block out of the next
    # block's vertical legs. Each block then needs at most K tracks.
    for k in range(1,len(ys)):
        passing=[ni for ni in range(len(rows)) if (ni,k-1) in at]
        capacity=len(tracks[k])
        if not capacity:continue
        for left,right in zip(passing,passing[1:]):
            if left//capacity==right//capacity:continue
            outgoing[at[left,k]].append((at[right,k-1],'order',None,None))
            outgoing[at[left,k-1]].append((at[right,k],'order',None,None))
    choice=[0]*len(ds);q=deque(range(len(ds)));queued=set(q);updates=0
    while q:
        i=q.popleft();queued.discard(i);v=ds[i][choice[i]]
        for j,kind,rch,iv in outgoing[i]:
            if kind=='order':
                c=bisect_left(ds[j][:,1],v[2]+lane-1e-9)
            else:
                strip=next(((l,h) for l,h in iv if l+GUARD<=v[0]<=h-GUARD),None)
                if strip is None:return {},dict(reason='no common strip')
                c=bisect_left(ds[j][:,0],max(v[0]-rch,strip[0]+GUARD)-1e-9)
            if c>=len(ds[j]):return {},dict(reason='capacity exhausted',net=rows[meta[j][0]]['net'],row=meta[j][1],updates=updates,
                cause=kind,from_net=rows[meta[i][0]]['net'],from_row=meta[i][1],value=v.tolist(),end=ds[j][-1].tolist(),corridor=iv)
            if c>choice[j]:
                choice[j]=c;updates+=1
                if j not in queued:q.append(j);queued.add(j)
    paths={}
    for ni,r in enumerate(rows):
        birth=min(k for n,k in at if n==ni);v=ds[at[ni,birth]][choice[at[ni,birth]]]
        paths[r['net']]=[(float(v[3]),float(r['y'])),(float(v[0]),ys[birth])]
    for k in range(1,len(ys)):
        jogs=[]
        for ni,r in enumerate(rows):
            if (ni,k-1) not in at:continue
            x0=ds[at[ni,k-1]][choice[at[ni,k-1]]][0];x1=ds[at[ni,k]][choice[at[ni,k]]][0]
            if abs(x0-x1)>1e-8:jogs.append((ni,x0,x1))
        tr=ordered_tracks(jogs,tracks[k],lane-GUARD/2) if jogs else {}
        if tr is None:return {},dict(reason='track certificate failed',row=k)
        for ni,x0,x1 in jogs:paths[rows[ni]['net']]+=[(float(x0),tr[ni]),(float(x1),tr[ni])]
    for r in rows:paths[r['net']].append(tuple(map(float,r['pad_xy'])))
    return paths,dict(updates=updates,variables=len(ds),slots=sum(len(d) for d in ds),
                      pads={r['net']:r['pad_num'] for r in rows}, refused=refused)


def source_order(rows, *geometry):
    """A lower source enters left of higher sources; split peers order by x."""
    ordered = sorted(rows, key=lambda r: (r['y'], min(r['vias']), r['net']))
    return {r['net']: i for i, r in enumerate(ordered)}, {}


def bridges(links, vias, F, a, s, py):
    """One direction of a parallel-notch bundle (down in these coordinates)."""
    aa,ss=a+GUARD,s+GUARD
    pieces={}
    for l in links:
        for side in ('A','B'):
            pix=tuple(sorted(l[side]))
            pieces[pix]=set(np.round(vias[list(pix),0],6))
    # Interval overlap components, computed before any copper.
    groups=[]
    for pix,xs in sorted(pieces.items(),key=lambda z:min(z[1])):
        if not groups or min(xs)>groups[-1]['hi']+1e-5:
            groups.append(dict(hi=max(xs),common=set(xs),pieces=[pix]))
        else:
            g=groups[-1];g['hi']=max(g['hi'],max(xs));g['common'] &= xs;g['pieces'].append(pix)
    group_of={p:i for i,g in enumerate(groups) for p in g['pieces']}
    if any(not g['common'] for g in groups):
        return {},{l['id']:'source group has no common via column' for l in links}
    levels=sorted({round(l['y'],6) for l in links},reverse=True)
    root=min(levels)-1.5*py-GUARD
    # Use the highest via-free track at or below the geometric notch root.
    yr=np.unique(vias[:,1]);near=yr[np.argmin(abs(yr-root))]
    if abs(near-root)<aa:
        root=float(near-aa)
    # A common via is insufficient: certify the complete strip, including
    # every reserved leg. End groups need only the side used by their links.
    extent=aa+(len(levels)-1)*ss
    for i,g in enumerate(groups):
        left=any(group_of[tuple(sorted(l['B']))]==i for l in links)
        right=any(group_of[tuple(sorted(l['A']))]==i for l in links)
        floor=root-(len(levels)-1)*ss
        ceiling=max(levels)-py/2-GUARD
        g['x']=next((x for x in sorted(g['common']) if F.covers(
            box(x-extent*left,floor,x+extent*right,ceiling))),None)
    if any(g['x'] is None for g in groups):
        return {},{l['id']:'no complete common strip' for l in links}
    paths,refused={},{}
    for l in links:
        ga=group_of[tuple(sorted(l['A']))];gb=group_of[tuple(sorted(l['B']))]
        if gb != ga+1:
            refused[l['id']]='link skips a source group';continue
        rank=levels.index(round(l['y'],6));off=aa+(len(levels)-1-rank)*ss
        xa,xb=groups[ga]['x'],groups[gb]['x'];y=l['y'];bottom=root-rank*ss
        shoulder=y-py/2-GUARD
        paths[l['id']]=[(xa,y),(xa,shoulder),(xa+off,shoulder),(xa+off,bottom),
                        (xb-off,bottom),(xb-off,shoulder),(xb,shoulder),(xb,y)]
    return paths,refused

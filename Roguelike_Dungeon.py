"""
Roguelike Dungeon — Adaptive Final v5 (Full Features Restored)
--------------------------------------------------------------
- 保留&恢复：地板格子背景、环绕地图、出生安全区、击退、粒子、受击闪红、屏幕震动、敌人子弹、死亡&GameOver、后台生成下一关
- 难度只影响敌人数量与速度；钥匙固定 3~4 把（离散，至少 2 把）
- HUD 显示 Difficulty 实时数值
"""

import math, random, time, gc, threading, queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame

# ---------------- Resource Manager（可用于未来资源/纹理统一管理） ----------------
class ResourceManager:
    def __init__(self):
        self._map = {}
    def acquire(self, key: str, creator):
        if key in self._map:
            surf, ref = self._map[key]
            self._map[key] = (surf, ref + 1)
            return surf
        surf = creator()
        self._map[key] = (surf, 1)
        return surf
    def release(self, key: str):
        if key not in self._map: return
        surf, ref = self._map[key]
        ref -= 1
        if ref <= 0:
            del self._map[key]
        else:
            self._map[key] = (surf, ref)

# ---------------- 常量 ----------------
TILE = 30
GW, GH = 40, 30
SCR_W, SCR_H = GW * TILE, GH * TILE
FPS = 60

T_WALL, T_FLOOR, T_DOOR_LOCK, T_DOOR_OPEN = 0, 1, 2, 3
SAFE_RADIUS_TILES = 6
ATTACK_COOLDOWN = 0.5

COL_BG = (12,12,16)
COL_FLOOR = (60,62,72)
COL_WALL = (24,26,32)
COL_GRID = (85,88,98)
COL_DOOR_L = (150,150,150)
COL_DOOR_O = (60,210,80)
COL_KEY = (0,255,100)
COL_WPN_MELEE = (255,230,90)
COL_WPN_RANGED = (120,200,255)
COL_PLAYER = (120,190,255)
COL_PLAYER_OUT = (220,240,255)
COL_PLAYER_HURT = (255,70,70)
COL_EN_CHASER = (235,80,80)
COL_EN_SHOOTER = (255,120,120)
COL_EN_TANK = (190,110,70)
COL_BULLET_ENEMY = (255,120,120)

SCORE_KILL = {"chaser":10,"shooter":12,"tank":20}
SCORE_KEY = 5
SCORE_CLEAR = 50

# 攻击（圆形范围，带 alpha）
A_MELEE = (255,230,90,160)
A_RANGED = (120,200,255,140)
R_MELEE = int(1.2*TILE)
R_RANGED = int(2.5*TILE)

# ---------------- 自适应难度（只作用于敌人数量&速度） ----------------
@dataclass
class Perf:
    elapsed: float
    damage_taken: int
    hp_end: int
    hp_max: int
    score_delta: int
    level_idx: int

class DiffManager:
    def __init__(self, base=1.0):
        self.diff = base
        self.recent_avg_score = 30.0
    def update(self, perf: Perf):
        speed_score = min(1.5, 90.0 / max(15.0, perf.elapsed))
        hp_score = perf.hp_end / max(1, perf.hp_max)
        if self.recent_avg_score <= 0: self.recent_avg_score = 30.0
        score_ratio = max(0.5, min(1.5, (perf.score_delta+1)/(self.recent_avg_score+1)))
        perf_score = 0.4*hp_score + 0.4*speed_score + 0.2*score_ratio
        target = 1.0 + (perf_score - 1.0)*0.9
        self.diff = max(1.0, min(12.0, self.diff*0.6 + target*0.8))
        self.recent_avg_score = self.recent_avg_score*0.7 + perf.score_delta*0.3
        return self.diff
    def speed_scale(self):
        # 放大敌人移动速度（限制上限，避免过快）
        return max(0.95, min(1.6, 1.0 + 0.10*self.diff))

# ---------------- 点位离散工具（防止物品/钥匙集中） ----------------
def spread_points(points, avoid=set(), min_dist=6, max_pick=5):
    result=[]
    random.shuffle(points)
    for p in points:
        if p in avoid: continue
        if all((p[0]-x)**2+(p[1]-y)**2>=min_dist**2 for x,y in result):
            result.append(p)
            if len(result)>=max_pick: break
    return result

# ---------------- 地牢生成 ----------------
def carve_tile(tiles,x,y,w=1):
    for yy in range(y - (w//2), y + (w//2) + 1):
        for xx in range(x - (w//2), x + (w//2) + 1):
            if 0<=xx<GW and 0<=yy<GH:
                tiles[yy][xx]=T_FLOOR

def make_dungeon(seed, idx, diff, enemy_mult=1.0):
    rnd=random.Random(seed)
    tiles=[[T_WALL for _ in range(GW)] for _ in range(GH)]
    rooms=[]
    for _ in range(rnd.randint(10,14)):
        w,h=rnd.randint(5,9),rnd.randint(4,7)
        x,y=rnd.randint(1,GW-w-2),rnd.randint(1,GH-h-2)
        r=pygame.Rect(x,y,w,h)
        if all(not r.colliderect(o.inflate(-2,-2)) for o in rooms):
            rooms.append(r)
            for yy in range(r.y,r.y+r.h):
                for xx in range(r.x,r.x+r.w):
                    tiles[yy][xx]=T_FLOOR
    if not rooms:
        r=pygame.Rect(GW//4,GH//4,8,6); rooms=[r]
        for yy in range(r.y,r.y+r.h):
            for xx in range(r.x,r.x+r.w): tiles[yy][xx]=T_FLOOR

    centers=[(r.centerx,r.centery) for r in rooms]
    connected={0}; edges=[]
    # 最小生成树式连接
    for i in range(1,len(centers)):
        nx,ny=centers[i]
        j=min(connected,key=lambda k:(centers[k][0]-nx)**2+(centers[k][1]-ny)**2)
        edges.append((i,j)); connected.add(i)
    # 额外随机连接
    for _ in range(max(2,int(len(rooms)*random.uniform(0.3,0.5)))):
        a,b=rnd.sample(range(len(rooms)),2)
        if a!=b: edges.append((a,b))
    # L型走廊，宽 1~3
    for i,j in edges:
        ra,rb=rooms[i],rooms[j]
        ax,ay=rnd.randint(ra.x+1,ra.x+ra.w-2),rnd.randint(ra.y+1,ra.y+ra.h-2)
        bx,by=rnd.randint(rb.x+1,rb.x+rb.w-2),rnd.randint(rb.y+1,rb.y+rb.h-2)
        w=rnd.randint(1,3)
        x,y=ax,ay
        while x!=bx: carve_tile(tiles,x,y,w); x+=1 if bx>x else -1
        while y!=by: carve_tile(tiles,x,y,w); y+=1 if by>y else -1
        carve_tile(tiles,x,y,w)

    rooms_sorted=sorted(rooms,key=lambda r:(r.y,r.x))
    start_g=(rooms_sorted[0].centerx,rooms_sorted[0].centery)

    # BFS 可达
    from collections import deque
    vis=[[False]*GW for _ in range(GH)]
    q=deque([start_g]); vis[start_g[1]][start_g[0]]=True; order=[start_g]
    while q:
        x,y=q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<GW and 0<=ny<GH and not vis[ny][nx] and tiles[ny][nx]==T_FLOOR:
                vis[ny][nx]=True; q.append((nx,ny)); order.append((nx,ny))

    # 门放在最远可达房间内部
    far_room=max(rooms,key=lambda r:(r.centerx-start_g[0])**2+(r.centery-start_g[1])**2)
    cand=[(x,y) for y in range(far_room.y+1,far_room.y+far_room.h-1)
                  for x in range(far_room.x+1,far_room.x+far_room.w-1) if vis[y][x]]
    door_g=cand[-1] if cand else order[-1]; tiles[door_g[1]][door_g[0]]=T_DOOR_LOCK

    sx,sy=start_g; safe_r2=SAFE_RADIUS_TILES**2
    def in_safe(px,py): return (px-sx)**2+(py-sy)**2<=safe_r2
    reachable=set(order)

    # 钥匙固定（3~4），与难度无关；保证至少 2 把；离散分布
    keys_count = random.randint(3,4)
    spread_base_keys=[p for p in reachable if p not in (start_g,door_g) and not in_safe(*p)]
    key_spawns = spread_points(spread_base_keys, avoid=set(), min_dist=6, max_pick=keys_count)
    if len(key_spawns) < 2:
        extra=[p for p in reachable if p not in (start_g,door_g)]
        random.shuffle(extra)
        key_spawns += extra[:2 - len(key_spawns)]

    # 武器离散分布
    wcount=random.randint(2,4)
    spread_base_weapons=[p for p in reachable if p not in (start_g,door_g) and not in_safe(*p)]
    wpos=spread_points(spread_base_weapons, avoid=set(key_spawns), min_dist=5, max_pick=wcount)
    if not wpos: wpos=[(sx+1, sy)]
    weapons=[(x,y,"melee" if i%2==0 else "ranged") for i,(x,y) in enumerate(wpos)]

    # 敌人数量由难度控制
    ecand=[p for p in reachable if p not in (start_g,door_g) and not in_safe(*p)]
    random.shuffle(ecand)
    base=6 + diff*3 + idx*0.5
    enemy_mult = max(0.6, 0.9 + 0.12*diff)
    count=int(base * enemy_mult)
    enemies=[(x,y,random.choices(["chaser","shooter","tank"],weights=[3,2,1])[0]) for x,y in ecand[:max(0,count)]]

    keys_required=len(key_spawns)  # 实际生成的钥匙数；收集完即可开门
    return tiles,start_g,door_g,key_spawns,weapons,enemies,keys_required

# ---------------- 特效 ----------------
class Particle:
    def __init__(self, x, y, vx, vy, life, color, size):
        self.x,self.y=x,y; self.vx,self.vy=vx,vy
        self.life=life; self.life_max=life; self.color=color; self.size=size; self.dead=False
    def update(self,dt):
        self.x+=self.vx*dt; self.y+=self.vy*dt
        self.life-=dt; self.dead=self.life<=0
    def draw(self,sc):
        t=max(0.0,self.life/self.life_max); a=int(220*t)
        col=(*self.color[:3],a)
        s=pygame.Surface((SCR_W,SCR_H),pygame.SRCALPHA)
        pygame.draw.circle(s,col,(int(self.x),int(self.y)),max(1,int(self.size*t)))
        sc.blit(s,(0,0))

class CircleAoE:
    def __init__(self, cx, cy, radius, tint_rgba, dmg=1, life=0.2):
        self.cx,self.cy = cx,cy
        self.radius = radius
        self.base_tint = tint_rgba   # (r,g,b,a)
        self.life = life
        self.life_max = life
        self.dmg = dmg
        self.dead = False
        self.hit_done = False
    def update(self, dt, level):
        if not self.hit_done:
            r2 = self.radius*self.radius
            hit_any = False
            for e in list(level.enemies):
                if e.dead: continue
                dx,dy = e.x - self.cx, e.y - self.cy
                if dx*dx + dy*dy <= r2:
                    e.hit(self.dmg, level, center=(self.cx, self.cy))
                    # 命中粒子
                    for _ in range(8):
                        ang=random.random()*math.tau
                        sp=random.uniform(80,180)
                        vx,vy=math.cos(ang)*sp,math.sin(ang)*sp
                        level.particles.append(Particle(e.x,e.y,vx,vy,0.25,(255,255,255),int(0.2*TILE)))
                    hit_any=True
            if hit_any:
                level.shake(0.12, amplitude=6)
            self.hit_done = True
        self.life -= dt
        self.dead = self.life <= 0
    def draw(self, sc):
        t = max(0.0, self.life / self.life_max)
        a = int(self.base_tint[3] * t)
        col = (self.base_tint[0], self.base_tint[1], self.base_tint[2], a)
        s = pygame.Surface((SCR_W,SCR_H), pygame.SRCALPHA)
        pygame.draw.circle(s, col, (int(self.cx), int(self.cy)), self.radius)
        sc.blit(s, (0,0))

class DeathFx:
    def __init__(self, x, y, color, size, life=0.2):
        self.x,self.y = x,y
        self.color = color
        self.size = size
        self.life = life
        self.life_max = life
        self.dead = False
    def update(self, dt):
        self.life -= dt
        self.dead = self.life <= 0
    def draw(self, sc):
        t = max(0.0, self.life / self.life_max)
        a = int(200 * t)
        col = (*self.color[:3], a)
        s = pygame.Surface((SCR_W,SCR_H), pygame.SRCALPHA)
        pygame.draw.circle(s, col, (int(self.x), int(self.y)), max(2, int(self.size*0.7)))
        sc.blit(s, (0,0))

# ---------------- 玩家 ----------------
class Player:
    def __init__(self,x,y):
        self.x,self.y=float(x),float(y)
        self.size= int(0.47*TILE)
        self.speed=220
        self.hp_max=8; self.hp=8; self.ifr=0
        self.melee_uses=0; self.ranged_uses=0; self.attack_cd=0
        self.damage_taken=0; self.keys=0
        self.hurt_flash=0.0
    def rect(self): return pygame.Rect(int(self.x)-self.size//2,int(self.y)-self.size//2,self.size,self.size)
    def move(self,dx,dy,dt,level):
        if dx==dy==0:return
        mag=math.hypot(dx,dy) or 1; vx=(dx/mag)*self.speed; vy=(dy/mag)*self.speed
        nx,ny=level.wrap_pos(self.x+vx*dt,self.y+vy*dt)
        rw=rh=self.size; rx=nx-rw/2; ry=self.y-rh/2
        if not level.rect_collides_wall(rx,ry,rw,rh): self.x=nx
        rx=self.x-rw/2; ry=ny-rh/2
        if not level.rect_collides_wall(rx,ry,rw,rh): self.y=ny
    def hit(self,d, level=None):
        if self.ifr>0:return
        self.hp=max(0,self.hp-d); self.damage_taken+=d; self.ifr=0.25
        self.hurt_flash=0.2
        if level: level.shake(0.12, amplitude=8)
    def can_attack(self): return self.attack_cd<=0 and (self.melee_uses>0 or self.ranged_uses>0)
    def try_attack(self,level):
        if not self.can_attack(): return
        self.attack_cd=ATTACK_COOLDOWN
        if self.melee_uses>0:
            self.melee_uses-=1
            level.aoes.append(CircleAoE(self.x, self.y, R_MELEE, A_MELEE, dmg=1))
        elif self.ranged_uses>0:
            self.ranged_uses-=1
            level.aoes.append(CircleAoE(self.x, self.y, R_RANGED, A_RANGED, dmg=1))
    def update(self,dt):
        if self.ifr>0:self.ifr-=dt
        if self.attack_cd>0:self.attack_cd-=dt
        if self.hurt_flash>0:self.hurt_flash-=dt
    def draw(self,sc):
        base_col = COL_PLAYER if self.hurt_flash<=0 else COL_PLAYER_HURT
        pygame.draw.circle(sc,base_col,(int(self.x),int(self.y)),self.size//2)
        pygame.draw.circle(sc,COL_PLAYER_OUT,(int(self.x),int(self.y)),self.size//2,2)

# ---------------- 敌人 ----------------
class Enemy:
    def __init__(self,x,y,hp,spd,size,col,etype):
        self.x,self.y=x,y; self.hp=hp; self.base_spd=spd; self.size=size; self.color=col; self.type=etype
        self.dead=False; self.cool=0
    def rect(self): return pygame.Rect(int(self.x)-self.size//2,int(self.y)-self.size//2,self.size,self.size)
    def speed(self,level):
        # 难度仅通过 diff_speed_scale 放大
        return (self.base_spd*0.5) * level.ai_scale * level.diff_speed_scale
    def _knockback(self, level, center, pixels=50):
        if center is None: return
        cx,cy = center
        dx,dy = self.x - cx, self.y - cy
        dist = math.hypot(dx,dy)
        if dist < 1e-3:
            ang = random.random()*math.tau
            dx,dy = math.cos(ang), math.sin(ang)
        else:
            dx,dy = dx/dist, dy/dist
        kb = pixels
        steps = max(1, int(kb / 6))
        step_px = kb / steps
        for _ in range(steps):
            nx = self.x + dx*step_px
            ny = self.y + dy*step_px
            nx, ny = (nx+SCR_W)%SCR_W, (ny+SCR_H)%SCR_H
            if level.in_safe_zone(nx, ny): break
            gx, gy = level.px_to_g(nx), level.py_to_g(ny)
            if level.tile_at(gx,gy) == T_WALL: break
            self.x, self.y = nx, ny
    def hit(self, d, level, center=None):
        kb = random.randint(40, 60)
        self._knockback(level, center, pixels=kb)
        if not self.dead:
            self.dead = True
            level.death_fx.append(DeathFx(self.x, self.y, self.color, self.size))
    def draw(self,sc): pygame.draw.rect(sc,self.color,self.rect(),border_radius=3)

class EnemyChaser(Enemy):
    def __init__(self,x,y): super().__init__(x,y,1,140,int(0.53*TILE),COL_EN_CHASER,"chaser")
    def update(self,dt,level):
        if self.dead or not level.enemies_active:return
        if level.in_safe_zone(self.x,self.y):return
        spd=self.speed(level); px,py=level.player.x,level.player.y
        dx,dy=px-self.x,py-self.y; d=math.hypot(dx,dy) or 1
        nx,ny=self.x+(dx/d)*spd*dt,self.y+(dy/d)*spd*dt
        if level.in_safe_zone(nx,ny): return
        if level.tile_at(level.px_to_g(nx),level.py_to_g(ny))!=T_WALL: self.x,self.y=nx,ny
        if self.rect().colliderect(level.player.rect()): level.player.hit(1, level)

class EnemyShooter(Enemy):
    def __init__(self,x,y): super().__init__(x,y,1,90,int(0.53*TILE),COL_EN_SHOOTER,"shooter"); self.cool=0
    def update(self,dt,level):
        if self.dead or not level.enemies_active:return
        if level.in_safe_zone(self.x,self.y):return
        self.cool-=dt; fire_cd=1.2/max(0.7,min(1.5,level.ai_scale))
        if self.cool<=0:
            self.cool=fire_cd; px,py=level.player.x,level.player.y
            ang=math.atan2(py-self.y,px-self.x); vx,vy=math.cos(ang)*300,math.sin(ang)*300
            level.spawn_enemy_bullet(self.x,self.y,vx,vy,1)

class EnemyTank(Enemy):
    def __init__(self,x,y): super().__init__(x,y,1,100,int(0.67*TILE),COL_EN_TANK,"tank")
    def update(self,dt,level):
        if self.dead or not level.enemies_active:return
        if level.in_safe_zone(self.x,self.y):return
        spd=self.speed(level)*0.8
        px,py=level.player.x,level.player.y
        dx,dy=px-self.x,py-self.y; d=math.hypot(dx,dy) or 1
        nx,ny=self.x+(dx/d)*spd*dt,self.y+(dy/d)*spd*dt
        if level.in_safe_zone(nx,ny): return
        if level.tile_at(level.px_to_g(nx),level.py_to_g(ny))!=T_WALL:self.x,self.y=nx,ny
        if self.rect().colliderect(level.player.rect()):level.player.hit(2, level)

# ---------------- 掉落 ----------------
class KeyPickup:
    def __init__(self,x,y): self.x,self.y=x,y; self.size=int(0.38*TILE); self.dead=False
    def rect(self): return pygame.Rect(self.x-self.size//2,self.y-self.size//2,self.size,self.size)
    def on_pick(self,player,game): self.dead=True; player.keys+=1; game.score+=SCORE_KEY
    def draw(self,sc): pygame.draw.rect(sc,COL_KEY,self.rect(),border_radius=4)

class WeaponPickup:
    def __init__(self,x,y,t): self.x,self.y=x,y; self.wtype=t; self.size=int(0.44*TILE); self.dead=False
    def rect(self): return pygame.Rect(self.x-self.size//2,self.y-self.size//2,self.size,self.size)
    def on_pick(self,p):
        self.dead=True
        if self.wtype == "melee": p.melee_uses += 3
        else: p.ranged_uses += 3
    def draw(self,sc):
        col=COL_WPN_MELEE if self.wtype=="melee" else COL_WPN_RANGED
        pygame.draw.rect(sc,col,self.rect(),border_radius=6)

# ---------------- 关卡 ----------------
class Level:
    def __init__(self,seed,idx,diff,rm,game):
        self.rm=rm; self.game=game
        # 敌人速度缩放：由 DiffManager 输出（难度→速度）
        self.diff_speed_scale = game.diffmgr.speed_scale()
        # 敌人数量乘子随难度变化；钥匙不受影响
        enemy_mult = max(0.6, 0.9 + 0.12*diff)

        (tiles,start_g,door_g,key_spawns,weapons,enemies,keys_required) = \
            make_dungeon(seed,idx,diff,enemy_mult=enemy_mult)

        self.data={"tiles":tiles,"start_g":start_g,"door_g":door_g,"key_spawns":key_spawns,
                   "weapon_spawns":weapons,"enemy_spawns":enemies,"keys_required":keys_required}
        sx,sy=start_g; self.spawn_px=sx*TILE+TILE//2; self.spawn_py=sy*TILE+TILE//2
        self.player=Player(self.spawn_px,self.spawn_py)

        # 敌人
        self.enemies=[]
        for gx,gy,t in enemies:
            px,py=gx*TILE+TILE//2,gy*TILE+TILE//2
            if t=="chaser":self.enemies.append(EnemyChaser(px,py))
            elif t=="shooter":self.enemies.append(EnemyShooter(px,py))
            else:self.enemies.append(EnemyTank(px,py))

        # 拾取物
        self.keys=[KeyPickup(x*TILE+TILE//2,y*TILE+TILE//2) for x,y in key_spawns]
        self.weapons=[WeaponPickup(x*TILE+TILE//2,y*TILE+TILE//2,wt) for x,y,wt in weapons]
        self.aoes=[]; self.enemy_bullets=[]
        self.death_fx=[]; self.particles=[]

        self.door_open=False; self.start_time=time.time()
        self.last_dir=(1,0); self.enemies_active=False
        self.ai_scale=1.0; self._ai_accum=0

        # 预渲染瓦片（地板带网格线）
        self.s_floor=self._mk_tile(COL_FLOOR,True)
        self.s_wall =self._mk_tile(COL_WALL ,False)
        self.s_doorL=self._mk_tile(COL_DOOR_L,True)
        self.s_doorO=self._mk_tile(COL_DOOR_O,True)

        # 屏幕震动控制
        self.shake_time=0.0; self.shake_amp=0

    def _mk_tile(self,col,grid=True):
        s=pygame.Surface((TILE,TILE)); s.fill(col)
        if grid:pygame.draw.rect(s,COL_GRID,s.get_rect(),1)
        return s.convert()

    # 震动
    def shake(self, dur, amplitude=6):
        self.shake_time = max(self.shake_time, dur)
        self.shake_amp = max(self.shake_amp, amplitude)

    # 坐标/碰撞辅助
    def wrap_pos(self,x,y): return (x+SCR_W)%SCR_W,(y+SCR_H)%SCR_H
    def px_to_g(self,px): return int(px//TILE)%GW
    def py_to_g(self,py): return int(py//TILE)%GH
    def tile_at(self,gx,gy): return self.data["tiles"][gy][gx]
    def rect_collides_wall(self,rx,ry,rw,rh):
        gx0=int(rx//TILE)%GW; gy0=int(ry//TILE)%GH
        gx1=int((rx+rw-1)//TILE)%GW; gy1=int((ry+rh-1)//TILE)%GH
        for gy in range(gy0,gy0+((rh//TILE)+2)):
            for gx in range(gx0,gx0+((rw//TILE)+2)):
                gxw,gyw=gx%GW,gy%GH
                if self.tile_at(gxw,gyw)==T_WALL:
                    tx,ty=gxw*TILE,gyw*TILE
                    if (rx<tx+TILE and rx+rw>tx and ry<ty+TILE and ry+rh>ty):return True
        return False
    def in_safe_zone(self,px,py):
        dx=(px-self.spawn_px)/TILE; dy=(py-self.spawn_py)/TILE
        return dx*dx+dy*dy<=SAFE_RADIUS_TILES**2

    # 敌人子弹
    class Bullet:
        def __init__(self,x,y,vx,vy,d): self.x,self.y=x,y; self.vx,self.vy=vx,vy; self.dmg=d; self.dead=False; self.r=int(0.1*TILE)
        def rect(self): return pygame.Rect(int(self.x)-self.r,int(self.y)-self.r,self.r*2,self.r*2)
    def spawn_enemy_bullet(self,x,y,vx,vy,d): self.enemy_bullets.append(Level.Bullet(x,y,vx,vy,d))

    # AI 激进度（轻微自适应，用于射手开火节奏/追击感）
    def _update_ai_scale(self,dt):
        self._ai_accum+=dt
        if self._ai_accum<1:return
        self._ai_accum=0; p=self.player
        hp_ratio=p.hp/max(1,p.hp_max)
        key_prog=(p.keys/self.data["keys_required"]) if self.data["keys_required"]>0 else 0
        strength=0.5*hp_ratio+0.5*min(1,key_prog)
        target=0.85+(strength-0.5)*0.7
        self.ai_scale=max(0.6,min(1.4,self.ai_scale*0.7+target*0.3))

    def update(self,dt):
        # 离开安全区后才激活敌人
        if not self.enemies_active and not self.in_safe_zone(self.player.x,self.player.y):self.enemies_active=True

        self._update_ai_scale(dt); self.player.update(dt)

        # AOE 更新
        for aoe in self.aoes: aoe.update(dt,self)
        self.aoes=[a for a in self.aoes if not a.dead]

        # 敌人更新 & 击杀计分
        alive=[]
        for e in self.enemies:
            e.update(dt,self)
            if e.dead:
                self.game.score+=SCORE_KILL.get(e.type,10)
            else:
                alive.append(e)
        self.enemies=alive

        # 敌人子弹
        for eb in self.enemy_bullets:
            eb.x+=eb.vx*dt; eb.y+=eb.vy*dt
            gx,gy=self.px_to_g(eb.x),self.py_to_g(eb.y)
            if self.tile_at(gx,gy)==T_WALL:eb.dead=True
            elif eb.rect().colliderect(self.player.rect()):self.player.hit(1, self);eb.dead=True
        self.enemy_bullets=[b for b in self.enemy_bullets if not b.dead]

        # 拾取
        for k in self.keys:
            if not k.dead and k.rect().colliderect(self.player.rect()):k.on_pick(self.player,self.game)
        self.keys=[k for k in self.keys if not k.dead]
        for w in self.weapons:
            if not w.dead and w.rect().colliderect(self.player.rect()):w.on_pick(self.player)
        self.weapons=[w for w in self.weapons if not w.dead]

        # 门：必须拿满“本关实际生成的钥匙数”才开启
        if not self.door_open and self.player.keys>=self.data["keys_required"]:
            dx,dy=self.data["door_g"];self.data["tiles"][dy][dx]=T_DOOR_OPEN;self.door_open=True

        # FX
        for fx in self.death_fx: fx.update(dt)
        self.death_fx=[f for f in self.death_fx if not f.dead]
        for p in self.particles: p.update(dt)
        self.particles=[pp for pp in self.particles if not pp.dead]

        # 震动计时
        if self.shake_time>0: self.shake_time-=dt
        if self.shake_time<=0: self.shake_time=0; self.shake_amp=0

    def draw(self,sc):
        # 震动偏移
        ox=oy=0
        if self.shake_time>0:
            amp=self.shake_amp
            ox=random.randint(-amp,amp); oy=random.randint(-amp,amp)

        # 瓦片（地板+网格）
        for y,row in enumerate(self.data["tiles"]):
            for x,t in enumerate(row):
                surf={T_WALL:self.s_wall,T_FLOOR:self.s_floor,T_DOOR_LOCK:self.s_doorL,T_DOOR_OPEN:self.s_doorO}[t]
                sc.blit(surf,(x*TILE+ox,y*TILE+oy))
        # 安全圈
        pygame.draw.circle(sc,(90,105,130),(int(self.spawn_px)+ox,int(self.spawn_py)+oy),SAFE_RADIUS_TILES*TILE,1)

        # 物件
        for k in self.keys:
            r=k.rect().move(ox,oy); pygame.draw.rect(sc,COL_KEY,r,border_radius=4)
        for w in self.weapons:
            col=COL_WPN_MELEE if w.wtype=="melee" else COL_WPN_RANGED
            r=w.rect().move(ox,oy); pygame.draw.rect(sc,col,r,border_radius=6)

        # 敌人 & 子弹
        for e in self.enemies:
            r=e.rect().move(ox,oy); pygame.draw.rect(sc,e.color,r,border_radius=3)
        for b in self.enemy_bullets:
            pygame.draw.circle(sc,COL_BULLET_ENEMY,(int(b.x)+ox,int(b.y)+oy),b.r)

        # AOE/死亡FX/粒子
        for a in self.aoes:
            s=pygame.Surface((SCR_W,SCR_H),pygame.SRCALPHA)
            t=a.life/a.life_max if a.life_max>0 else 0
            aa=int(a.base_tint[3]*max(0,t))
            col=(a.base_tint[0],a.base_tint[1],a.base_tint[2],aa)
            pygame.draw.circle(s,col,(int(a.cx)+ox,int(a.cy)+oy),a.radius)
            sc.blit(s,(0,0))
        for fx in self.death_fx:
            s=pygame.Surface((SCR_W,SCR_H),pygame.SRCALPHA)
            t=max(0.0,fx.life/fx.life_max); a=int(200*t)
            col=(*fx.color[:3],a)
            pygame.draw.circle(s,col,(int(fx.x)+ox,int(fx.y)+oy),max(2,int(fx.size*0.7)))
            sc.blit(s,(0,0))
        for p in self.particles:
            s=pygame.Surface((SCR_W,SCR_H),pygame.SRCALPHA)
            tt=max(0.0,p.life/p.life_max); a=int(220*tt)
            col=(*p.color[:3],a)
            pygame.draw.circle(s,col,(int(p.x)+ox,int(p.y)+oy),max(1,int(p.size*tt)))
            sc.blit(s,(0,0))

        # 玩家
        base_col = COL_PLAYER if self.player.hurt_flash<=0 else COL_PLAYER_HURT
        pygame.draw.circle(sc,base_col,(int(self.player.x)+ox,int(self.player.y)+oy),self.player.size//2)
        pygame.draw.circle(sc,COL_PLAYER_OUT,(int(self.player.x)+ox,int(self.player.y)+oy),self.player.size//2,2)

    def try_enter_door(self):
        if not self.door_open:return False
        dx,dy=self.data["door_g"];r=pygame.Rect(dx*TILE,dy*TILE,TILE,TILE)
        return self.player.rect().colliderect(r)

    def release(self):
        self.enemies.clear();self.keys.clear();self.weapons.clear();self.enemy_bullets.clear()
        self.aoes.clear();self.death_fx.clear();self.particles.clear();self.data["tiles"].clear();gc.collect()

# ---------------- 后台关卡生成 ----------------
class GenWorker:
    def __init__(self):
        self.q=queue.Queue();self.out=queue.Queue()
        threading.Thread(target=self._run,daemon=True).start()
    def request(self,seed,idx,diff,rm,game):self.q.put((seed,idx,diff,rm,game))
    def try_get(self):
        try:return self.out.get_nowait()
        except queue.Empty:return None
    def _run(self):
        while True:
            seed,idx,diff,rm,game=self.q.get()
            try:
                lvl=Level(seed,idx,diff,rm,game)
                self.out.put(lvl)
            except Exception as e:
                print("Thread Error:", e)

# ---------------- 游戏 ----------------
class Game:
    def __init__(self):
        pygame.init()
        self.sc=pygame.display.set_mode((SCR_W,SCR_H))
        pygame.display.set_caption("Roguelike Dungeon — Adaptive Final v5")
        self.clock=pygame.time.Clock();self.font=pygame.font.SysFont(None,22)
        self.rm=ResourceManager();self.worker=GenWorker();self.diffmgr=DiffManager(1.0)
        self.seed=random.randrange(1_000_000);self.level_idx=1
        self.level=Level(self.seed,self.level_idx,self.diffmgr.diff,self.rm,self)
        self.worker.request(self.seed+1,self.level_idx+1,self.diffmgr.diff,self.rm,self)
        self.pending=None;self.last_move=(1,0);self.game_over=False;self.score=0
        self.score_last_level=0  # 记录上一关累计分，用于计算 score_delta

    def _hud(self):
        p=self.level.player
        txt=f"SCORE {self.score} | HP {p.hp}/{p.hp_max} | Keys {p.keys}/{self.level.data['keys_required']} | Melee {p.melee_uses} | Range {p.ranged_uses} | Lv {self.level_idx} | AI {self.level.ai_scale:.2f} | Diff {self.diffmgr.diff:.2f} | FPS {int(self.clock.get_fps())}"
        t=self.font.render(txt,True,(240,240,240));self.sc.blit(t,(8,6))
        tip=self.font.render("Move: Arrows/WASD (wrap) | Attack: Space (Circle AoE; stacks) | Collect ALL green keys -> Door -> Next",True,(210,210,210))
        self.sc.blit(tip,(8,28))

    def _game_over(self):
        self.sc.fill((20,0,0));f1=pygame.font.SysFont(None,72);f2=pygame.font.SysFont(None,28)
        t1=f1.render("GAME OVER",True,(255,80,80));self.sc.blit(t1,(SCR_W//2-t1.get_width()//2,SCR_H//2-100))
        t2=f2.render(f"Final Score: {self.score}",True,(240,220,220));self.sc.blit(t2,(SCR_W//2-t2.get_width()//2,SCR_H//2))
        pygame.display.flip()

    def swap_to_next(self):
        # 计算本关表现并更新难度
        cur_time = time.time()-self.level.start_time
        cur_delta = self.score - self.score_last_level
        perf=Perf(cur_time, self.level.player.damage_taken, self.level.player.hp, self.level.player.hp_max, cur_delta, self.level_idx)
        self.diffmgr.update(perf); self.score+=SCORE_CLEAR
        self.score_last_level = self.score  # 更新基线

        # 取后台新关（若暂未准备好则主线程生成以避免等待太久）
        new = self.pending
        if not new:
            wait_start = time.time()
            while not new and time.time() - wait_start < 3.0:
                new = self.worker.try_get()
                if not new: time.sleep(0.05)
            if not new:
                print("⚠️ Background gen slow, building next level on main thread.")
                new = Level(self.seed+1, self.level_idx+1, self.diffmgr.diff, self.rm, self)

        # 释放旧关
        self.level.release();del self.level;gc.collect()

        # 切换
        self.level_idx+=1; self.level=new; self.seed+=1
        self.worker.request(self.seed+1,self.level_idx+1,self.diffmgr.diff,self.rm,self)
        self.pending=None

    def run(self):
        running=True
        while running:
            dt=self.clock.tick(FPS)/1000
            for e in pygame.event.get():
                if e.type==pygame.QUIT:running=False
                elif e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_ESCAPE:running=False
                    elif not self.game_over and e.key==pygame.K_SPACE:self.level.player.try_attack(self.level)

            if self.game_over:
                self._game_over();continue

            k=pygame.key.get_pressed()
            dx=int(k[pygame.K_RIGHT]or k[pygame.K_d])-int(k[pygame.K_LEFT]or k[pygame.K_a])
            dy=int(k[pygame.K_DOWN]or k[pygame.K_s])-int(k[pygame.K_UP]or k[pygame.K_w])
            self.level.player.move(dx,dy,dt,self.level)
            if dx or dy:self.last_move=(dx,dy)
            self.level.last_dir=self.last_move

            self.level.update(dt)
            if self.level.player.hp<=0:self.game_over=True;continue

            if not self.pending:
                nxt=self.worker.try_get()
                if nxt:self.pending=nxt

            if self.level.try_enter_door():self.swap_to_next()

            self.sc.fill(COL_BG);self.level.draw(self.sc);self._hud();pygame.display.flip()
        pygame.quit()


# ---------------- Persistent Difficulty Display (右上角实时难度显示) ----------------
def add_persistent_difficulty_display(GameClass):
    """
    给 Game 类添加右上角实时难度显示功能。
    调用方法：add_persistent_difficulty_display(Game)
    """

    # 保留原 HUD 绘制逻辑
    old_hud = GameClass._hud

    def new_hud(self, *a, **kw):
        # 执行原 HUD
        old_hud(self, *a, **kw)

        # 绘制右上角难度显示（常驻）
        diff_text = f"Difficulty: {self.diffmgr.diff:.2f}x"
        txt = self.font.render(diff_text, True, (255, 220, 90))
        bg = pygame.Surface((txt.get_width()+14, txt.get_height()+8), pygame.SRCALPHA)
        bg.fill((40, 40, 50, 180))  # 半透明背景
        x = SCR_W - bg.get_width() - 12
        y = 8
        self.sc.blit(bg, (x, y))
        self.sc.blit(txt, (x+7, y+4))

    GameClass._hud = new_hud

# 注入到 Game 类
add_persistent_difficulty_display(Game)


if __name__=="__main__":
    Game().run()

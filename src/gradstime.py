import datetime as dt

class GradsTime(object):

    def __init__(self, idate, itime, shift_dt=0):

        idate = str(idate)
        itime = '%06d'%(itime,)

        self.idt = dt.datetime.strptime(idate+itime,'%Y%m%d%H%M%S')
        self.idt += dt.timedelta(hours=shift_dt)

    def strftime(self, s, hours=0, minutes=0):

        return self.strvtime(self.stritime(s), hours, minutes)

    def strvtime(self, s, hours=0, minutes=0):

        vdt = self.idt + dt.timedelta(hours=hours, minutes=minutes)

        tokens = { '%y2' : vdt.strftime('%y'),
                   '%y4' : vdt.strftime('%Y'),
                   '%m1' : str(vdt.month),
                   '%m2' : vdt.strftime('%m'),
                   '%mc' : vdt.strftime('%b'),
                   '%d1' : str(vdt.day),
                   '%d2' : vdt.strftime('%d'),
                   '%h1' : str(vdt.hour),
                   '%h2' : vdt.strftime('%H'),
                   '%h3' : '%03d'%(hours,),
                   '%f2' : '%02d'%(hours,),
                   '%f3' : '%03d'%(hours,),
                   '%n2' : vdt.strftime('%M'),
                   '%s2' : vdt.strftime('%S'),
                   '%j3' : vdt.strftime('%j')
                 }

        for k,v in iter(tokens.items()): s = s.replace(k,v)

        return s

    def stritime(self, s, hours=None, minutes=None):

        tokens = { '%iy2' : self.idt.strftime('%y'),
                   '%iy4' : self.idt.strftime('%Y'),
                   '%im1' : str(self.idt.month),
                   '%im2' : self.idt.strftime('%m'),
                   '%imc' : self.idt.strftime('%b'),
                   '%id1' : str(self.idt.day),
                   '%id2' : self.idt.strftime('%d'),
                   '%ih1' : str(self.idt.hour),
                   '%ih2' : self.idt.strftime('%H'),
                   '%ih3' : '%03d'%(self.idt.hour,),
                   '%in2' : self.idt.strftime('%M'),
                   '%is2' : self.idt.strftime('%S'),
                   '%ij3' : self.idt.strftime('%j')
                 }

        for k,v in iter(tokens.items()): s = s.replace(k,v)

        return s

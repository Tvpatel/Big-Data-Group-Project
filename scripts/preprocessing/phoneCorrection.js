conn = new Mongo();
db = conn.getDB("weisheng");

db.hygiene.find({'PHONE':{$type:18}}).forEach(
	function(e) {db.hygiene.update({_id:e._id}, {$set:{PHONE: ""+e.PHONE}})}
	);
db.hygiene.find({'PHONE':{$type:16}}).forEach(
	function(e) {db.hygiene.update({_id:e._id}, {$set:{PHONE: e.PHONE.toString()}})}
	);
db.hygiene.find({'PHONE':{$exists:true}}).forEach(
	function(e) {db.hygiene.update({_id:e._id}, {$set:{PHONE: "+1"+e.PHONE}})}
);

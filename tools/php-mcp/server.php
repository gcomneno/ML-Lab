<?php

declare(strict_types=1);

function readStdin(): array {
    $input = stream_get_contents(STDIN);
    if ($input === false || $input === '') return [];
    $json = json_decode($input, true);
    return is_array($json) ? $json : [];
}

function writeJson(array $payload): void {
    fwrite(STDOUT, json_encode($payload, JSON_UNESCAPED_SLASHES) . PHP_EOL);
}

$request = readStdin();
$type = $request['type'] ?? '';

$tools = [
  'ping' => [
    'desc' => 'Risponde "pong".',
    'schema' => ['type'=>'object','properties'=>[]],
    'fn' => fn(array $args) => ['message'=>'pong']
  ],
  'sum' => [
    'desc' => 'Somma due numeri: a + b.',
    'schema' => [
      'type'=>'object',
      'properties'=>['a'=>['type'=>'number'],'b'=>['type'=>'number']],
      'required'=>['a','b']
    ],
    'fn' => function(array $args){ 
      $a=$args['a']??null; $b=$args['b']??null;
      if(!is_numeric($a)||!is_numeric($b)) return ['error'=>'Argomenti non numerici'];
      return ['result'=>(float)$a+(float)$b];
    }
  ],
];

switch ($type) {
  case 'initialize':
    $list=[];
    foreach($tools as $name=>$meta){
      $list[]=['name'=>$name,'description'=>$meta['desc'],'input_schema'=>$meta['schema']];
    }
    writeJson(['type'=>'initialize_result','tools'=>$list]); break;

  case 'call_tool':
    $name=$request['name']??''; $args=is_array($request['args']??null)?$request['args']:[];
    if(!isset($tools[$name])){ writeJson(['type'=>'error','error'=>"Tool '$name' non trovato"]); break; }
    try { $result=$tools[$name]['fn']($args);
      writeJson(['type'=>'tool_result','name'=>$name,'result'=>$result]);
    } catch(Throwable $e){ writeJson(['type'=>'error','error'=>$e->getMessage()]); }
    break;

  default:
    writeJson(['type'=>'error','error'=>'Richiesta non valida o type mancante']); break;
}
